# ml-coding-practice

Daily hands-on practice with repeated use of earlier concepts. English explanations and examples live directly in each exercise's Python docstrings. DSA practice is separate.

## Practice log

- Day 1 — completed: NumPy basics, slicing, masks, axes, reshape, and simple broadcasting. 29 tests passed at completion.
- Day 2 — next: [gradebook.py](week01_numpy_logreg/day02_gradebook/gradebook.py). Process one shared dataset, then combine your functions into a report. Eight core functions (~40 minutes) and one optional exercise.

The direction remains NumPy → models from scratch → PyTorch and integrated ML coding. Later exercises reuse earlier skills; the daily pace follows actual progress.

## Setup

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Day 2

Read and implement one function at a time in `gradebook.py`:

```bash
python -m pytest -q week01_numpy_logreg/day02_gradebook -k course_means
# All core exercises (skip the optional needs_support exercise):
python -m pytest -q week01_numpy_logreg/day02_gradebook -k 'not needs_support'
# Both days, including optional work:
python -m pytest -q week01_numpy_logreg
```

New starter functions intentionally raise `NotImplementedError`. Tests for unfinished functions will fail. Day 1 contains your completed work; Day 2 contains no solution implementations.
