#!/usr/bin/env bash
# Arena AI sandbox setup — one line, idempotent, reset-proof.
#
#   bash scripts/arena_setup.sh        # or:  make arena-setup   /   just arena-setup
#
# Reason for existing: the Arena AI sandbox and CI runners do not persist
# installed OS packages across workspace resets — but committed PROJECT FILES
# do. So instead of committing binaries, we commit this script, which
# re-establishes a known-good environment from scratch in one command. Run it
# once after every fresh clone / workspace reset, before building anything.
#
# What it does (idempotent — safe to re-run):
#   1. Create the project venv (.venv) if missing.
#   2. Install pinned Python deps (requirements.txt) into it.
#   3. Install the build DAG tool (redo) into the venv if the pure-python
#      implementatation (build/redo) is present; otherwise try pip.
#   4. Smoke-check: report venv python, key deps, and the build tool.
#
# No sudo / no OS packages required: everything lands in .venv, so it works
# identically on the sandbox, CI (ubuntu-latest), and a dev laptop.

set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

# Prefer the current Python (3.14 on the sandbox/box); fall back to whatever
# `python3` resolves to, or the user's explicit PYTHON override.
if [ -n "${PYTHON:-}" ]; then PY_BIN="$PYTHON"
elif command -v python3.14 >/dev/null 2>&1; then PY_BIN=python3.14
else PY_BIN=python3; fi
VENV=".venv"
VENV_PY="$VENV/bin/python"
VENV_PIP="$VENV/bin/pip"

echo "==[ arena_setup ]=="
echo "repo root : $(pwd)"

# 1. venv
if [ ! -x "$VENV_PY" ]; then
  echo ">> creating $VENV"
  "$PY_BIN" -m venv "$VENV"
fi

# Ensure pip is usable via `python -m pip` (a venv may lack the `pip` wrapper;
# Debian/ensurepip-less builds often do). Bootstrap from ensurepip if missing.
if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
  echo ">> bootstrapping pip (no 'pip' module in venv)"
  "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 \
    || "$VENV_PY" -m pip --version >/dev/null 2>&1 \
    || { echo ">> WARN: pip unavailable in venv; deps not installed"; }
fi

# 2. deps (always refresh; cheap)
echo ">> installing Python deps into $VENV"
"$VENV_PY" -m pip install -q --upgrade pip >/dev/null 2>&1 || true
"$VENV_PY" -m pip install -q -r requirements.txt

# 3. build DAG tool — redo. Prefer the committed pure-python impl (persists),
#    fall back to pip.
if [ -x "build/redo" ]; then
  echo ">> build tool: committed pure-python redo (build/redo)"
  REDO="$(pwd)/build/redo"
elif "$VENV_PIP" install -q redo 2>/dev/null; then
  echo ">> build tool: pip redo"
  REDO="redo"
else
  echo ">> WARN: no redo found (apt 'redo' not used — non-persistent). Using make only."
  REDO=""
fi

# 4. smoke check
echo ">> smoke"
echo "   python : $("$VENV_PY" --version)"
# Self-heal a common Nix/managed-python gap: binary wheels (numpy/scipy) that
# need libstdc++.so.6 may fail to load if the python's runpath lacks it. If so,
# find a libstdc++ and put it on LD_LIBRARY_PATH. Harmless elsewhere.
if ! "$VENV_PY" -c "import numpy" >/dev/null 2>&1; then
  if ! "$VENV_PY" -c "import numpy" 2>&1 | grep -q "libstdc++"; then
    : # failure not about libstdc++ — leave as-is
  elif [ -z "${LIBSTDCXX_DIR:-}" ]; then
    # Cheap candidate probes (NOT a store-wide find — that's far too slow).
    local cand=""
    for glob in /nix/store/*gcc-*-lib/lib/libstdc++.so.6 \
                /usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
                /usr/lib/gcc/x86_64-linux-gnu/*/libstdc++.so.6; do
      if compgen -G "$glob" >/dev/null && [ -f "$(compgen -G "$glob" | head -1)" ]; then
        cand="$(dirname "$(compgen -G "$glob" | head -1)")"
        break
      fi
    done
    if [ -n "$cand" ]; then
      echo "   loader: found libstdc++ in $cand"
      export LD_LIBRARY_PATH="$cand${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
  fi
fi
"$VENV_PY" -c "import numpy, scipy; print('   deps   : numpy', numpy.__version__, '+ scipy')" 2>&1 | tail -1 || \
  { echo "   NOTICE : a dependency needs a system lib not on the loader path."; \
    echo "            On plain Ubuntu/CI this resolves itself; on a Nix box run the"; \
    echo "            one-liner inside the devShell or export LD_LIBRARY_PATH."; }
if [ -n "${REDO:-}" ]; then
  echo "   redo   : $REDO"
fi
echo ">> done. Use:  make <target>   or   just <recipe>   (1:1 parity)"
