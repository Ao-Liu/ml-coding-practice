"""Day 20 — One sales table, three roles, one complete pipeline (~40–45 minutes).

Connect earlier skills without adding a new formula. Rows are already ordered
from oldest to newest. Use the earliest rows for training, the next rows for
validation, and the remaining rows for test. Each target belongs to the same
row as its sales. This is a simple chronological exercise, not random splitting.
All arrays are finite. Each set has at least one row; split counts are valid.
Do not change inputs. Exercises 1–4 repeat direct NumPy; reuse them in exercise 5.
"""
from typing import Tuple
import numpy as np


def split_days(sales: np.ndarray, actual: np.ndarray, n_train: int, n_validation: int
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """1 — Review slicing: give the same rows the same role (~7 minutes).

    Learn:
    sales is (N,D), actual is (N,), with days in matching chronological order.
    Take the first n_train days for training, the NEXT n_validation days for
    validation, and all remaining days for test. n_validation is a COUNT,
    not an ending position: validation ends at n_train + n_validation.
    Apply identical slices to sales and actual so features keep their targets.
    Return six arrays in pairs: train sales/targets, validation sales/targets,
    test sales/targets. Keep 2D feature tables even when a split has one row.
    Views are fine for this exercise; callers must not modify the returned data.

    Independent API reminder:
        values = np.array([10,20,30,40,50])
        print(values[1:3])  # [20,30]: start included, stop excluded
        print(values[3:])   # [40,50]

    Task: Return chronological training, validation, and test feature/target pairs.
    Example input/output:
        sales = np.array([[1,7], [3,7], [2,7], [4,7]])
        actual = np.array([1.,5.,3.,7.])
        n_train = 2
        n_validation = 1
        Expected output: (np.array([[1,7],[3,7]]), np.array([1.,5.]),
                          np.array([[2,7]]), np.array([3.]),
                          np.array([[4,7]]), np.array([7.]))

    Pause: what goes wrong if sales and actual use different split boundaries?
    """
    return (sales[0:n_train], actual[0:n_train],
            sales[n_train:n_train+n_validation], actual[n_train:n_train+n_validation],
            sales[n_train+n_validation:], actual[n_train+n_validation:])


def prepare_three_sets(train_sales: np.ndarray, validation_sales: np.ndarray,
                       test_sales: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2 — Review: the training ruler transforms all three sets (~8 minutes).

    Learn:
    Compute column means and population std from train_sales only (axis=0,
    ddof=0). Replace exactly zero std with 1, keeping real means unchanged.
    Apply (sales - means) / safe_scales separately to all three tables using
    those same statistics. Return only the three transformed tables today.
    Each keeps its own row count and the common D columns. Targets are absent
    because revenue is not standardized. Validation/test values cannot change
    the ruler or transformed training rows, even if their values are extreme.
    A product constant in training may vary later: do not force its later
    transformed values to zero. Training mean 7 and safe scale 1 map 9 to 2.

    Independent API reminder:
        values = np.array([3.,-2.,8.])
        values[values < 0.] = 0.
        print(values)  # [3.,0.,8.]

    Task: Return training, validation, and test features standardized using training statistics only.
    Example input/output:
        train_sales = np.array([[1,7], [3,7]])
        validation_sales = np.array([[2,7]])
        test_sales = np.array([[4,7], [1,7], [2,9]])
        Expected output: (np.array([[-1.,0.],[1.,0.]]), np.array([[0.,0.]]),
                          np.array([[2.,0.],[-1.,0.],[0.,2.]]))

    Pause: does the test set need its own mean and std just because it has more days?
    """
    scale = np.std(train_sales, axis=0, ddof=0)
    scale[scale == 0] = 1
    train_means = np.mean(train_sales, axis=0)



    return (train_sales - train_means) / scale, (validation_sales - train_means) / scale, (test_sales - train_means) / scale


def fit_training_rows(features: np.ndarray, actual: np.ndarray, weights: np.ndarray,
                      bias: float, learning_rate: float, steps: int
                      ) -> Tuple[np.ndarray, float]:
    """3 — Review the training loop using training rows only (~10 minutes).

    Learn:
    features is already standardized TRAINING data (N,D). Copy weights so
    input arrays stay unchanged; return independent weights even for steps=0.
    For exactly steps updates, compute predictions = features @ weights + bias
    and signed errors = predictions - actual using the current parameters.
    Recall dw = (2/N) * (features.T @ errors), db = 2 * mean(errors).
    N counts training days. Compute both gradients from the same errors, then
    subtract learning_rate times each gradient from the respective parameter.
    Recompute errors as the parameters change. No validation or test data
    enter this function. A constant standardized column is zero, so that
    weight's gradient is zero; the other parameters can still change.
    With zero steps return starting values, keeping weights independently owned.

    Independent API reminder:
        original = np.array([2.,5.])
        working = original.copy()
        working -= 1.
        print(original)  # [2.,5.]
        print(working)   # [1.,4.]

    Task: Return the fitted independent weight array and bias as a Python float.
    Example input/output:
        features = np.array([[-1.,0.], [1.,0.]])
        actual = np.array([1.,5.])
        weights = np.array([0.,0.])
        bias = 0.
        learning_rate = 0.1
        steps = 2
        Expected output: (np.array([0.72,0.]), 1.08)

    Pause: should N include the validation or test days?
    """
    weights = weights.copy()
    pred = features @ weights + bias
    signed_err = pred - actual
    n_days = features.shape[0]

    for _ in range(steps):
        dw = (2 / n_days) * (features.T @ signed_err)
        db = 2 * np.mean(signed_err)
        weights = weights - (dw * learning_rate)
        bias -= (db * learning_rate)
        pred = features @ weights + bias
        signed_err = pred - actual

    return weights, float(bias)




def test_comparison(test_predictions: np.ndarray, test_actual: np.ndarray,
                    train_actual: np.ndarray) -> Tuple[float, float, float]:
    """4 — Review baseline: choose on train, score on test (~6 minutes).

    Learn:
    The model predictions are already supplied, one per TEST day. Compute
    their MSE against test_actual. The baseline guesses mean(train_actual)
    for each of those same test days, and is also scored against test_actual.
    Training revenues determine what to guess, not which days to score.
    A scalar baseline can broadcast directly against the test target array.
    If you build a repeated array, its length must be the TEST day count;
    training and test arrays need not have the same length.
    Return baseline test MSE, model test MSE, and baseline minus model MSE.
    Keep negative improvement. Never compare a training MSE with a test MSE
    to claim which predictor is better on test: use the same evaluation days.

    Independent API reminder:
        answers = np.array([2.,5.,8.])
        print(4. - answers)  # [2.,-1.,-4.]

    Task: Return baseline test MSE, model test MSE, and their baseline-minus-model difference as floats.
    Example input/output:
        test_predictions = np.array([7.,1.,3.])
        test_actual = np.array([7.,1.,3.])
        train_actual = np.array([1.,5.])
        Expected output: (20/3, 0.0, 20/3)

    Pause: why does a training target array of length 2 still support scoring 3 test days?
    """
    train_mean = np.mean(train_actual)
    baseline_pred = np.full(test_actual.shape[0], train_mean, dtype=float)

    b_signed_err = baseline_pred - test_actual
    b_mse = np.mean(b_signed_err ** 2)

    t_signed_err = test_predictions - test_actual
    t_mse = np.mean(t_signed_err ** 2)

    return float(b_mse), float(t_mse), float(b_mse - t_mse)


def run_pipeline(sales: np.ndarray, actual: np.ndarray, n_train: int, n_validation: int,
                 weights: np.ndarray, bias: float, learning_rate: float, steps: int
                 ) -> Tuple[float, float, float, float]:
    """5 — Connect your functions into one small experiment (~12 minutes).

    Learn:
    Reuse exercises 1–4 here. Split the chronological rows with matching targets,
    standardize all three feature tables using training statistics, and fit
    the supplied model only on training rows. Then freeze the fitted parameters.
    Compute validation predictions and MSE against validation targets. Separately
    compute test predictions and MSE against test targets. Each prediction uses
    the corresponding standardized feature table with the same fitted weights.
    Finally compare the model with the training-mean baseline on TEST days.

    Return validation model MSE, test model MSE, baseline test MSE, and test
    improvement (baseline test MSE minus model test MSE), all Python floats.
    There is only one candidate today; validation reports its score without
    selecting among models. Learning rate and steps are supplied, not tuned by
    this function. No argmin or extra update on validation/test rows is needed.
    Changing held-out targets must not change fitted parameters. With steps=0,
    still split, preprocess, and evaluate using the starting model parameters.

    Independent API reminder:
        pair = (np.array([2.,4.]), 1.0)
        values, offset = pair
        print(values + offset)  # [3.,5.]

    Task: Return validation model MSE, test model MSE, baseline test MSE, and test improvement.
    Example input/output:
        sales = np.array([[1,7], [3,7], [2,7], [4,7], [1,7], [2,9]])
        actual = np.array([1.,5.,3.,7.,1.,3.])
        n_train = 2
        n_validation = 1
        weights = np.array([0.,0.])
        bias = 0.
        learning_rate = 0.5
        steps = 1
        Expected output: (0.0, 0.0, 20/3, 20/3)

    Pause: if only the final three target values change, should validation MSE change?
    """
    train_sales = sales[:n_train]
    validation_sales = sales[n_train:n_train + n_validation]
    test_sales = sales[n_train + n_validation:]

    train_means = np.mean(train_sales, axis=0)
    train_scale = np.std(train_sales, axis=0, ddof=0)
    train_scale[train_scale == 0] = 1
    train_stand = (train_sales - train_means) / train_scale

    validation_stand = (validation_sales - train_means) / train_scale

    test_stand = (test_sales - train_means) / train_scale

    train_actual = actual[:n_train]

    weights = weights.copy()
    train_pred = train_stand @ weights + bias
    train_signed_err = train_pred - train_actual

    for _ in range(steps):
        dw = (2 / n_train) * (train_stand.T @ train_signed_err)
        db = 2 * np.mean(train_signed_err)
        weights = weights - (dw * learning_rate)
        bias -= (db * learning_rate)
        train_pred = train_stand @ weights + bias
        train_signed_err = train_pred - train_actual

    validation_pred = validation_stand @ weights + bias
    v_mse = np.mean((validation_pred - actual[n_train:n_train + n_validation]) ** 2)

    test_model_pred = test_stand @ weights + bias
    t_m_mse = np.mean((test_model_pred - actual[n_train + n_validation:]) ** 2)

    train_means = np.mean(train_actual)
    test_baseline_pred = np.full(len(actual) - n_train - n_validation, train_means, dtype=float)
    t_b_mse = np.mean((test_baseline_pred - actual[n_train + n_validation:]) ** 2)

    return float(v_mse), float(t_m_mse), float(t_b_mse), float(t_b_mse - t_m_mse)

