#!/usr/bin/env bash
# Run only the tests for modules changed in the working tree / staged.
# Uses the mirror naming convention tests/test_<module>.py -> models/<module>.py.
#
# Usage:
#   bash scripts/test_changed.sh            # unstaged + staged changes
#   bash scripts/test_changed.sh <base>     # changes vs <base> (e.g. origin/main)
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

base="${1:-}"
if [ -n "$base" ]; then
  changed=$(git diff --name-only "$base" 2>/dev/null | sort -u)
else
  changed=$( { git diff --name-only; git diff --cached --name-only; } 2>/dev/null | sort -u )
fi

if [ -z "$changed" ]; then
  echo "No changed files. Nothing to test."
  exit 0
fi

tests=()
for f in $changed; do
  case "$f" in
    tests/*_test.py|tests/test_*.py)
      tests+=("$f");;
    models/*.py)
      base_mod=$(basename "$f" .py)
      if [ -f "tests/test_${base_mod}.py" ]; then
        tests+=("tests/test_${base_mod}.py")
      fi
      ;;
  esac
done

if [ ${#tests[@]} -eq 0 ]; then
  echo "Changed files have no mirror test files (checked: $changed)."
  exit 0
fi

echo "Running ${#tests[@]} test file(s):"
printf '  %s\n' "${tests[@]}"
exec python -m pytest "${tests[@]}" -q
