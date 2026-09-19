# ml-coding-practice

Daily hands-on practice with repeated use of earlier concepts. English explanations and examples live directly in each exercise's Python docstrings. DSA practice is separate.

## Practice log

- Day 1 — completed: NumPy basics, slicing, masks, axes, reshape, and simple broadcasting. 29 tests passed at completion.
- Day 2 — completed: [gradebook.py](week01_numpy_logreg/day02_gradebook/gradebook.py). Process one shared dataset, then combine your functions into a report. All 22 tests passed at completion.

- Day 3 — completed: [sales_practice.py](week01_numpy_logreg/day03_sales/sales_practice.py). Sales analysis, broadcasting, and matrix-vector multiplication; all 17 tests passed.

- Day 4 — completed: [revenue_evaluation.py](week01_numpy_logreg/day04_evaluation/revenue_evaluation.py). Signed errors, MAE, MSE, and model evaluation; all 21 tests passed.

- Day 5 — implemented with guidance: [gradient_practice.py](week01_numpy_logreg/day05_gradient_step/gradient_practice.py). All 21 tests passed; gradient intuition needs reinforcement.

The direction remains NumPy → models from scratch → PyTorch and integrated ML coding. Later exercises reuse earlier skills; the daily pace follows actual progress.

## Setup

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Day 6 — completed: understand one bias update (17 tests passed)

Open [bias_practice.py](week01_numpy_logreg/day06_bias_intuition/bias_practice.py): five small exercises (~35–45 minutes) using one product. Observe how bias and weight affect predictions, compare three bias choices by MSE, interpret the bias slope, and make one bias update. Detailed Learn sections are inside the functions. There is no transpose, matrix multiplication, or training loop today.

```bash
# Start with one exercise:
python -m pytest -q week01_numpy_logreg/day06_bias_intuition -k change_bias
# All Day 6 exercises:
python -m pytest -q week01_numpy_logreg/day06_bias_intuition
# All days:
python -m pytest -q week01_numpy_logreg
```

Days 1–6 contain completed implementations. All 127 tests passed at Day 6 completion. Day 6 consolidated bias changes, gradient direction, and a single update.

## Exercise format

Use connected scenarios with repeated hands-on NumPy practice, minimizing calls to earlier exercise functions. Keep function type annotations, but omit verbose Inputs/Output blocks from future docstrings. Explain necessary shapes naturally in Learn or example comments. Keep Task to one sentence and show complete, self-contained input/output examples.

Write more detailed English Learn sections inside each function: explain the purpose and meaning of symbols, walk through a small numerical reasoning example, and introduce only one conceptual change at a time. Give independent NumPy API examples, not complete solutions disguised with different numbers. Do not assume that showing a formula teaches its meaning. Use rectangular, non-symmetric examples when explaining transpose.

Leave new implementations blank. Discuss each next day before generating it. Prioritize understanding and repetition over keeping the original daily schedule. Day 6 should revisit parameter effects, gradient direction, and a single update before adding training loops or more matrix calculus.
