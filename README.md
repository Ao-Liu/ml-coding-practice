# ml-coding-practice

Daily hands-on practice with repeated use of earlier concepts. English explanations and examples live directly in each exercise's Python docstrings. DSA practice is separate.

## Practice log

- Day 1 — completed: NumPy basics, slicing, masks, axes, reshape, and simple broadcasting. 29 tests passed at completion.
- Day 2 — completed: [gradebook.py](week01_numpy_logreg/day02_gradebook/gradebook.py). Process one shared dataset, then combine your functions into a report. All 22 tests passed at completion.

- Day 3 — completed: [sales_practice.py](week01_numpy_logreg/day03_sales/sales_practice.py). Sales analysis, broadcasting, and matrix-vector multiplication; all 17 tests passed.

- Day 4 — completed: [revenue_evaluation.py](week01_numpy_logreg/day04_evaluation/revenue_evaluation.py). Signed errors, MAE, MSE, and model evaluation; all 21 tests passed.

- Day 5 — implemented with guidance: [gradient_practice.py](week01_numpy_logreg/day05_gradient_step/gradient_practice.py). All 21 tests passed; gradient intuition needs reinforcement.

- Day 6 — completed: [bias_practice.py](week01_numpy_logreg/day06_bias_intuition/bias_practice.py). Parameter effects, bias slope, and a single update; all 17 tests passed.

- Day 7 — completed: [bias_training.py](week01_numpy_logreg/day07_bias_training/bias_training.py). Repeated bias updates, loss history, and learning-rate comparisons; all 15 tests passed.

- Day 8 — completed: [linear_training.py](week02_linear_models/day08_weight_and_bias/linear_training.py). Learning one weight and bias together; all 17 tests passed.

- Day 9 — completed: [two_product_training.py](week02_linear_models/day09_two_products/two_product_training.py). Separate product gradients, joint updates, and training; all 16 tests passed.

- Day 10 — completed: [vector_practice.py](week02_linear_models/day10_vector_gradients/vector_practice.py). Vector predictions, transpose, gradients, and one joint update; all 16 tests passed.

- Day 11 — completed: [vector_training.py](week02_linear_models/day11_vector_training/vector_training.py). Vector training loops, independent weights, loss history, and error inspection; all 14 tests passed.

- Day 12 — completed: [validation_practice.py](week02_linear_models/day12_validation/validation_practice.py). Training/validation split, fixed-model evaluation, and error masks; all 13 tests passed after correcting a mask comparison typo.

- Day 13 — completed: [scaling_practice.py](week02_linear_models/day13_standardization/scaling_practice.py). Column statistics, standardization, and training-only preprocessing; all 12 tests passed.

- Day 14 — completed: [scaled_training.py](week02_linear_models/day14_scaled_training/scaled_training.py). Standardized model training, raw-sales prediction, and validation; all 12 tests passed.

- Day 15 — completed: [constant_features.py](week03_linear_models/day15_constant_features/constant_features.py). Constant columns, safe scales, saved statistics, and training; all 16 tests passed.

- Day 16 — completed: [baseline_practice.py](week03_linear_models/day16_baseline/baseline_practice.py). Training-mean baseline, validation MSE, and model improvement; all 16 tests passed.

- Day 17 — completed: [validation_report.py](week03_linear_models/day17_validation_report/validation_report.py). Baseline/model predictions, daily errors, MSE reduction, and validation reports; all 14 tests passed.

- Day 18 — completed: [model_selection.py](week03_linear_models/day18_model_selection/model_selection.py). Candidate MSEs, argmin, selected rows, and baseline comparisons; all 15 tests passed.

- Day 19 — completed: [heldout_practice.py](week03_linear_models/day19_heldout_test/heldout_practice.py). Validation selection, fixed-model test scoring, and baseline comparison; all 16 tests passed after correcting baseline length for unequal training/test sizes.

- Day 20 — completed: [sales_pipeline.py](week03_linear_models/day20_sales_pipeline/sales_pipeline.py). Chronological splitting, train-only preprocessing/fitting, and complete evaluation; all 15 tests passed.

- Day 21 — completed: [binary_days.py](week03_linear_models/day21_binary_labels/binary_days.py). Binary labels, probability thresholds, accuracy, and daily error masks; all 16 tests passed.

- Day 22 — completed: [sigmoid_practice.py](week04_logistic_regression/day22_sigmoid/sigmoid_practice.py). Linear scores, sigmoid probabilities, labels, and accuracy; all 15 tests passed.

The direction remains NumPy → models from scratch → PyTorch and integrated ML coding. Later exercises reuse earlier skills; the daily pace follows actual progress.

## Setup

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Day 23 — completed: true-class probability and binary cross entropy

Open [probability_loss.py](week04_logistic_regression/day23_probability_loss/probability_loss.py): five exercises (~35–40 minutes). Practice p versus 1-p, select the true-class probability, learn negative log with numeric examples, average into BCE, and compare accuracy with loss. No gradients or training yet. Detailed English teaching and full examples are in the function docstrings.

```bash
python -m pytest -q week04_logistic_regression/day23_probability_loss -k negative_probabilities
python -m pytest -q week04_logistic_regression/day23_probability_loss
# All days:
python -m pytest -q week01_numpy_logreg week02_linear_models week03_linear_models week04_logistic_regression
```

Days 1–23 are completed. All 381 tests passed at Day 23 completion.

## Exercise format

Schedule a larger comprehensive pipeline exercise periodically, rather than every day; keep intervening lessons small and progressive.

Use connected scenarios with repeated hands-on NumPy practice, minimizing calls to earlier exercise functions. Keep function type annotations, but omit verbose Inputs/Output blocks from future docstrings. Explain necessary shapes naturally in Learn or example comments. Keep Task to one sentence and show complete, self-contained input/output examples.

Write more detailed English Learn sections inside each function: explain the purpose and meaning of symbols, walk through a small numerical reasoning example, and introduce only one conceptual change at a time. Give independent NumPy API examples, not complete solutions disguised with different numbers. Do not assume that showing a formula teaches its meaning. Use rectangular, non-symmetric examples when explaining transpose.

Leave new implementations blank. Discuss each next day before generating it. Prioritize understanding and repetition over keeping the original daily schedule. Day 6 should revisit parameter effects, gradient direction, and a single update before adding training loops or more matrix calculus.
