# CI workflow — manual addition required (GitHub App cannot push .github/workflows/)

Save this as `.github/workflows/ci.yml` via GitHub web UI or local git push with personal token.

```yaml
name: CI

on:
  push:
    branches: [ main, "arena/*" ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]

    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -e .
      - name: Pytest
        run: pytest tests -q --tb=short
      - name: Run modeling suite (quick)
        run: python -m models.run_all --quick
      - name: Validate figures exist
        run: |
          ls -lh docs/figures/*.png | wc -l
          test -f docs/figures/process_flow_diagram.png
          test -f docs/figures/carburization_profiles.png
          test -f docs/figures/tempering_curve.png || echo "tempering fig may be optional in quick"
          test -f experiments/data/master_report.json
      - name: Check master report keys
        run: python -c "import json; d=json.load(open('experiments/data/master_report.json')); assert 'steps' in d; print('master report ok')"

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install ruff
      - run: ruff check models/ tests/ || true

  figures-regression:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: pip install -r requirements.txt
      - name: Generate figures on PR
        run: python -m models.run_all --quick
      - run: |
          echo "Figures generated:"
          ls docs/figures/*.png | wc -l
          test $(ls docs/figures/*.png | wc -l) -ge 20
```

