# ml-coding-practice

Daily hands-on practice with repeated use of earlier concepts. English explanations and examples live directly in each exercise's Python docstrings. DSA practice is separate.

## Practice log

- Day 1 — completed: NumPy basics, slicing, masks, axes, reshape, and simple broadcasting. 29 tests passed at completion.
- Day 2 — completed: [gradebook.py](week01_numpy_logreg/day02_gradebook/gradebook.py). Process one shared dataset, then combine your functions into a report. All 22 tests passed at completion.

The direction remains NumPy → models from scratch → PyTorch and integrated ML coding. Later exercises reuse earlier skills; the daily pace follows actual progress.

## Setup

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Day 3 — completed (17 tests passed)

Open [sales_practice.py](week01_numpy_logreg/day03_sales/sales_practice.py): seven core exercises (~40–50 minutes) and one optional review, from shop sales calculations to `X @ w + b`. Type and shape contracts, short tasks, and examples are inside the functions. Repeat the NumPy operations directly rather than calling earlier exercise functions.

```bash
python -m pytest -q week01_numpy_logreg/day03_sales -k daily_revenue
# All core exercises (skip the optional center_sales exercise):
python -m pytest -q week01_numpy_logreg/day03_sales -k 'not center_sales'
# All days, including optional work:
python -m pytest -q week01_numpy_logreg
```

New starter functions intentionally raise `NotImplementedError`. Tests for unfinished functions will fail. Days 1–3 contain your completed work. All 68 tests passed at Day 3 completion.

## Exercise format

Use connected scenarios with repeated NumPy practice, minimizing calls to earlier exercise functions. Put English teaching notes in function docstrings: independent API examples (not disguised solutions), explicit input/output types and shapes, a one-sentence Task, and self-contained examples listing every input and expected output. Leave new implementations blank. Discuss each next day before generating it.
