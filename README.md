# ml-coding-practice

Small daily exercises: read an explanation, implement one function, then run its test. Exercise instructions are in English; Python syntax basics are skipped.

## Current practice

[Day 1: NumPy basics](week01_numpy_logreg/day01_numpy/README.md)

Start with creating arrays, reading shapes, and selecting columns. Then try elementwise arithmetic and row sums. Allow about 20–30 minutes, and stop after the first three exercises if you want more time with the basics.

The broader direction is NumPy → Logistic Regression. We will introduce reshaping and broadcasting before normalization, losses, gradients, and training. The pace will follow your progress rather than a fixed seven-day deadline.

## Getting started

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Start with just exercise 1:
python -m pytest -q week01_numpy_logreg/day01_numpy -k make_array
```

The starter functions intentionally raise `NotImplementedError`; each test will fail until you implement its function. No solution implementations are included.
