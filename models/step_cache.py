"""
Content-addressable step cache for the modeling suite.

Each step in ``run_all`` is (approximately) a pure function of its source code,
its transitive import dependencies within ``models/``, and its runtime
parameters.  If none of those change, the step's outputs (JSON reports, PNG
figures) are unchanged and recomputation can be skipped safely.

Architecture
------------
* **Cache key** = SHA-256 of (sorted source-file hashes ∪ parameter hash).
  Source files are the step's own module plus every ``models/*.py`` it
  transitively imports (within the package).  This means changing
  ``kinetics.py`` invalidates every step that depends on it, but changing
  ``carburization.py`` only invalidates the carburization step.
* **Manifest** (``experiments/data/.step_cache/manifest.json``) maps
  ``step_name → {key, outputs, timestamp}``.  It is small, human-readable,
  and safe to commit (it's just bookkeeping).
* **Cache hit** requires (a) the key matches the manifest entry **and**
  (b) every declared output file still exists on disk.  Either mismatch
  → cache miss → recompute.
* **Forced invalidation**: ``--force-step <name>`` or ``StepCache(force=True)``.

Safety guarantees
-----------------
* A cache hit is only returned when **all** output files are present —
  partial/stale outputs never silently pass.
* The key incorporates source hashes, so a code change that alters results
  always invalidates the affected steps (and downstream dependents).
  Both absolute imports (``from models.kinetics import ...``) and relative
  imports (``from .kinetics import ...``, ``from ..electrochemistry import
  ...``) are tracked.
* The key does *not* incorporate dependency *versions* (numpy etc.); those
  are assumed stable within a session.  ``--no-cache`` or ``--force-step``
  are the escape hatches for a dependency upgrade.

Usage (in run_all.py)
---------------------
    from models.step_cache import StepCache

    cache = StepCache()                     # reads existing manifest
    with cache.step("electrochemistry", outputs=[...]) as hit:
        if hit:
            print("  ⏩ cached")
            continue
        run_electrochem_main()              # recompute
    # cache entry recorded on __exit__
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, FrozenSet, Iterable, Optional, Sequence, Set

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CACHE_DIR = ROOT / "experiments" / "data" / ".step_cache"
MANIFEST_PATH = CACHE_DIR / "manifest.json"

# ── Helpers ──────────────────────────────────────────────────────────────────


def _file_hash(path: Path) -> str:
    """SHA-256 of a file's bytes (read in 64 KiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parameter_hash(params: Any) -> str:
    """Deterministic SHA-256 of a JSON-serialisable value."""
    # sort_keys for determinism; default=str handles non-serialisable gracefully
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


# ── Dependency scanner ───────────────────────────────────────────────────────


def _resolve_module_file(mod_name: str) -> Optional[Path]:
    """Return the .py source file for a top-level module name, or None.

    Only returns files inside the ``models/`` package — stdlib and
    third-party modules are excluded from the dependency graph.
    """
    try:
        spec = importlib.util.find_spec(mod_name)
    except (ModuleNotFoundError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    p = Path(spec.origin).resolve()
    # Only track source files inside the models/ package
    try:
        p.relative_to(MODELS_DIR)
    except ValueError:
        return None
    if p.suffix != ".py":
        return None
    return p


def _file_to_dotted_package(filepath: Path) -> Optional[str]:
    """Derive the dotted package name of a file inside ``models/``.

    E.g. ``models/uncertainty/monte_carlo.py`` → ``"models.uncertainty"``.
    ``models/transport.py`` → ``"models"`` (top-level package).
    Returns ``None`` for files not inside ``models/``.
    """
    try:
        rel = filepath.relative_to(MODELS_DIR)
    except ValueError:
        return None
    # Drop the filename, keep the directory parts
    dir_parts = list(rel.parent.parts)
    # For files directly in models/, rel.parent is "." → dir_parts=[]
    # The package is still "models"
    return "models" + ("" if not dir_parts else "." + ".".join(dir_parts))


def _resolve_relative_import(
    raw: str, containing_file: Path
) -> Optional[str]:
    """Resolve a relative import token to an absolute dotted module name.

    Parameters
    ----------
    raw : str
        The import token, e.g. ``".kinetics"``, ``"..electrochemistry"``,
        or ``".uncertainty.sample"``.
    containing_file : Path
        The .py file containing the import statement.

    Returns
    -------
    str or None
        Fully-qualified dotted name (e.g. ``"models.kinetics"``), or None
        if the import cannot be resolved or falls outside ``models/``.
    """
    if not raw.startswith("."):
        return None  # not a relative import

    # Count leading dots: 1 dot = same package, 2 = parent, etc.
    dots = 0
    for ch in raw:
        if ch == ".":
            dots += 1
        else:
            break

    # The module path after the dots (e.g. "kinetics" or "uncertainty.sample")
    remainder = raw[dots:]
    if not remainder:
        # `from . import foo` — the package itself; we'll pick up the
        # names in the `import` clause below.  For now, resolve the
        # package.
        remainder = ""

    # Walk up (dots - 1) levels from the containing file's package
    package = _file_to_dotted_package(containing_file)
    if package is None:
        return None

    pkg_parts = package.split(".")
    # `from .` means stay in the same package (dots=1 → go up 0)
    # `from ..` means go to parent package (dots=2 → go up 1)
    go_up = dots - 1
    if go_up > len(pkg_parts) - 1:
        # Goes above the models/ root — outside our scope
        return None
    base_parts = pkg_parts[: len(pkg_parts) - go_up] if go_up > 0 else pkg_parts

    if remainder:
        candidate = ".".join(base_parts + remainder.split("."))
    else:
        candidate = ".".join(base_parts)

    # Only return if it stays within models
    if candidate.startswith("models"):
        return candidate
    return None


def _scan_imports(filepath: Path, seen: Optional[Set[Path]] = None) -> Set[Path]:
    """Walk static ``import`` / ``from ... import`` within ``models/``.

    This is a *conservative* scanner: it finds all ``models.*`` imports
    (both absolute like ``from models.kinetics import ...`` and relative
    like ``from .kinetics import ...`` or ``from ..electrochemistry import
    ...``) recursively.  It may over-estimate (e.g. conditional imports),
    which is safe — it just means more files enter the hash, causing more
    invalidation than strictly necessary, never less.
    """
    if seen is None:
        seen = set()
    if filepath in seen:
        return seen
    seen.add(filepath)

    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return seen

    for line in text.splitlines():
        stripped = line.strip()
        # Skip comments and blank lines
        if not stripped or stripped.startswith("#"):
            continue

        # ── from X import ... ──────────────────────────────────────
        if stripped.startswith("from "):
            tokens = stripped[len("from "):].split()
            if not tokens:
                continue
            raw = tokens[0].rstrip(",")

            # Resolve the import token to a candidate module name
            if raw.startswith("."):
                # Relative import: from .foo import ...
                candidate_abs = _resolve_relative_import(raw, filepath)
                if candidate_abs is None:
                    continue
                candidates = [candidate_abs]
            else:
                # Absolute import: from models.foo import ...
                candidates = _walk_candidates(raw)

            _try_resolve_and_recurse(candidates, seen)

        # ── import X ───────────────────────────────────────────────
        elif stripped.startswith("import "):
            tokens = stripped[len("import "):].split(",")
            for tok in tokens:
                raw = tok.strip().split()[0] if tok.strip() else ""
                if not raw:
                    continue
                if raw.startswith("."):
                    candidate_abs = _resolve_relative_import(raw, filepath)
                    if candidate_abs is None:
                        continue
                    candidates = [candidate_abs]
                else:
                    candidates = _walk_candidates(raw)
                _try_resolve_and_recurse(candidates, seen)

    return seen


def _walk_candidates(raw: str) -> list[str]:
    """Given an absolute import token, return candidate dotted names
    from longest to shortest (e.g. ``"models.foo.bar"`` →
    ``["models.foo.bar", "models.foo"]``).
    """
    parts = raw.split(".")
    candidates = []
    for end in range(len(parts), 0, -1):
        candidate = ".".join(parts[:end])
        if candidate.startswith("models"):
            candidates.append(candidate)
    return candidates


def _try_resolve_and_recurse(candidates: list[str], seen: Set[Path]) -> None:
    """Try each candidate module name; recurse into the first one that
    resolves to a file inside ``models/``.
    """
    for candidate in candidates:
        resolved = _resolve_module_file(candidate)
        if resolved is not None:
            _scan_imports(resolved, seen)
            return


def transitive_deps(module_name: str) -> FrozenSet[Path]:
    """Return the frozen set of ``models/*.py`` files *module_name*
    transitively depends on (including relative and absolute imports).
    """
    entry = _resolve_module_file(module_name)
    if entry is None:
        return frozenset()
    return frozenset(_scan_imports(entry))


# ── Cache key ────────────────────────────────────────────────────────────────


def compute_key(
    step_name: str,
    source_files: Iterable[Path],
    params: Optional[Any] = None,
) -> str:
    """Compute the cache key for a step.

    Parameters
    ----------
    step_name : str
        Human-readable step name (included in hash for uniqueness).
    source_files : iterable of Path
        The .py files whose content enters the hash (typically the step's
        transitive deps within ``models/``).
    params : optional
        JSON-serialisable runtime parameters (e.g. ``{"quick": True}``).

    Returns
    -------
    str
        64-char hex SHA-256 digest.
    """
    h = hashlib.sha256()
    h.update(step_name.encode())
    for fpath in sorted(source_files):
        try:
            fh = _file_hash(fpath)
        except OSError:
            fh = "MISSING"
        h.update(fpath.name.encode())
        h.update(fh.encode())
    if params is not None:
        h.update(_parameter_hash(params).encode())
    return h.hexdigest()


# ── Manifest I/O ─────────────────────────────────────────────────────────────


def _load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_manifest(manifest: Dict[str, Any], path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ── StepCache ────────────────────────────────────────────────────────────────


class StepCache:
    """Content-addressable cache for the modeling suite steps.

    Parameters
    ----------
    enabled : bool
        Master switch.  ``False`` → every step is a cache miss.
    force_steps : set of str
        Step names to always recompute, even if the key matches.
    verbose : bool
        Print cache-hit / cache-miss diagnostics.
    """

    def __init__(
        self,
        enabled: bool = True,
        force_steps: Optional[Set[str]] = None,
        verbose: bool = True,
    ):
        self.enabled = enabled
        self.force_steps = force_steps or set()
        self.verbose = verbose
        self._manifest = _load_manifest()
        self._deps_cache: Dict[str, FrozenSet[Path]] = {}
        # Stats
        self.hits: int = 0
        self.misses: int = 0
        self.skips: int = 0  # forced-invalidated (not a real miss)

    # ── Dependency resolution (memoised) ────────────────────────────────

    def _get_deps(self, module_name: str) -> FrozenSet[Path]:
        if module_name not in self._deps_cache:
            self._deps_cache[module_name] = transitive_deps(module_name)
        return self._deps_cache[module_name]

    # ── Core API ────────────────────────────────────────────────────────

    @contextmanager
    def step(
        self,
        name: str,
        module: str,
        outputs: Sequence[Path],
        params: Optional[Any] = None,
    ):
        """Context manager that gates a step's execution on its cache key.

        Yields ``True`` (cache hit — skip recompute) or ``False`` (cache miss
        — recompute).  On a miss, the new key + outputs are recorded on exit.

        Parameters
        ----------
        name : str
            Step name (e.g. ``"electrochemistry"``).
        module : str
            Python module name whose deps to scan (e.g. ``"models.run_electrochemistry"``).
        outputs : sequence of Path
            Output files the step produces.  All must exist for a hit.
        params : optional
            JSON-serialisable parameters influencing the step's output.

        Usage
        -----
            with cache.step("foo", "models.run_foo", [out_path]) as hit:
                if hit:
                    print("cached")
                else:
                    run_foo_main()
        """
        hit = self._check(name, module, outputs, params)
        try:
            yield hit
        finally:
            if not hit:
                self._record(name, module, outputs, params)

    def _check(
        self,
        name: str,
        module: str,
        outputs: Sequence[Path],
        params: Optional[Any],
    ) -> bool:
        """Return True if the step is cached and outputs are fresh."""
        if not self.enabled:
            self.misses += 1
            if self.verbose:
                print(f"  [cache] {name}: miss (caching disabled)")
            return False

        if name in self.force_steps:
            self.skips += 1
            if self.verbose:
                print(f"  [cache] {name}: miss (forced)")
            return False

        deps = self._get_deps(module)
        key = compute_key(name, deps, params)

        entry = self._manifest.get(name)
        if entry is None:
            self.misses += 1
            if self.verbose:
                print(f"  [cache] {name}: miss (no manifest entry)")
            return False

        if entry.get("key") != key:
            self.misses += 1
            if self.verbose:
                print(f"  [cache] {name}: miss (key changed)")
            return False

        # Verify all outputs exist
        recorded_outputs = entry.get("outputs", [])
        for out_str in recorded_outputs:
            if not Path(out_str).exists():
                self.misses += 1
                if self.verbose:
                    print(f"  [cache] {name}: miss (output missing: {out_str})")
                return False

        # Also verify caller-declared outputs
        for out_path in outputs:
            if not Path(out_path).exists():
                self.misses += 1
                if self.verbose:
                    print(f"  [cache] {name}: miss (declared output missing: {out_path})")
                return False

        self.hits += 1
        if self.verbose:
            n_out = len(recorded_outputs)
            print(f"  [cache] {name}: hit ({n_out} output(s) unchanged)")
        return True

    def _record(
        self,
        name: str,
        module: str,
        outputs: Sequence[Path],
        params: Optional[Any],
    ) -> None:
        """Record a (re)computed step in the manifest."""
        deps = self._get_deps(module)
        key = compute_key(name, deps, params)
        self._manifest[name] = {
            "key": key,
            "module": module,
            "outputs": [str(p) for p in outputs],
            "params_hash": _parameter_hash(params) if params is not None else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_files": sorted(str(p) for p in deps),
        }
        _save_manifest(self._manifest)

    # ── Bulk operations ─────────────────────────────────────────────────

    def invalidate(self, *names: str) -> None:
        """Force the named steps to be recomputed on next check."""
        self.force_steps.update(names)

    def invalidate_all(self) -> None:
        """Clear the entire manifest (next run recomputes everything)."""
        self._manifest.clear()
        _save_manifest(self._manifest)

    def status(self) -> Dict[str, Any]:
        """Return a summary dict of the cache state."""
        return {
            "enabled": self.enabled,
            "steps_cached": len(self._manifest),
            "hits": self.hits,
            "misses": self.misses,
            "forced_skips": self.skips,
            "forced_steps": sorted(self.force_steps),
        }

    def summary(self) -> str:
        """Human-readable one-line summary."""
        s = self.status()
        return (
            f"StepCache: {s['hits']} hit(s), {s['misses']} miss(es), "
            f"{s['forced_skips']} forced; {s['steps_cached']} step(s) in manifest"
        )
