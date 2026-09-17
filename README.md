# ml-coding-practice

Daily hands-on practice with repeated use of earlier concepts. English explanations and examples live directly in each exercise's Python docstrings. DSA practice is separate.

## Practice log

- Day 1 — completed: NumPy basics, slicing, masks, axes, reshape, and simple broadcasting. 29 tests passed at completion.
- Day 2 — completed: [gradebook.py](week01_numpy_logreg/day02_gradebook/gradebook.py). Process one shared dataset, then combine your functions into a report. All 22 tests passed at completion.

- Day 3 — completed: [sales_practice.py](week01_numpy_logreg/day03_sales/sales_practice.py). Sales analysis, broadcasting, and matrix-vector multiplication; all 17 tests passed.

The direction remains NumPy → models from scratch → PyTorch and integrated ML coding. Later exercises reuse earlier skills; the daily pace follows actual progress.

## Setup

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Day 4 — completed: prediction errors and evaluation (21 tests passed)

Open [revenue_evaluation.py](week01_numpy_logreg/day04_evaluation/revenue_evaluation.py): seven exercises (~40–50 minutes), from repeated prediction to signed errors, MAE, MSE, filtering, model comparison, and a complete evaluation report. All teaching notes and complete examples are in the function docstrings. Implement NumPy operations directly in each function.

```bash
# Start with one function:
python -m pytest -q week01_numpy_logreg/day04_evaluation -k predict_revenue
# All Day 4 exercises:
python -m pytest -q week01_numpy_logreg/day04_evaluation
# All days:
python -m pytest -q week01_numpy_logreg
```

Days 1–4 contain your completed work. All 89 tests passed at Day 4 completion.

## Exercise format

Use connected scenarios with repeated NumPy practice, minimizing calls to earlier exercise functions. Put English teaching notes in function docstrings: independent API examples (not disguised solutions), explicit input/output types and shapes, a one-sentence Task, and self-contained examples listing every input and expected output. Leave new implementations blank. Discuss each next day before generating it.
