"""Day 12 — Does the shop model work on days it did not train on?

Five exercises, about 35–45 minutes. The only new idea is separating data
used for parameter updates from data used to check the resulting model.
The prediction, MSE, and gradient formulas are unchanged.

sales rows are days in chronological order; columns are products.
actual contains one observed revenue per row. We train on earlier days
and validate on later days. This is a small time-ordered exercise, not a
claim that one split or a few observations guarantee future performance.

All inputs are valid and finite. Sales are nonnegative integer arrays
(N,D); targets and weights are float arrays (N,) and (D,). Splits are
nonempty, products have the same order in both sets, and steps >= 0.
learning_rate > 0; intermediate values stay finite. Do not modify inputs.
Use Python floats for scalar results, float arrays for predictions, and
integer arrays for selected IDs. Work directly in each function rather
than calling earlier exercises. Only exercise 4 needs a training-step loop.

From the repository root:
  python -m pytest -q week02_linear_models/day12_validation -k split_days
Unfinished exercises intentionally raise NotImplementedError.
"""
from signal import signal
from typing import Tuple
import numpy as np


def split_days(sales: np.ndarray, actual: np.ndarray, train_days: int
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """1 — Keep some later days out of training (~6 minutes).

    Learn:
    Training loss measures fit to examples used to adjust parameters.
    To check other examples, reserve some rows BEFORE training. Here the
    first train_days rows form the training set; the remaining rows form
    the validation set. Assume 1 <= train_days < the total number of days.

    Each sales row and its actual revenue belong together. Split both
    arrays at the SAME row boundary and preserve their chronological order.
    For example, keeping the first 3 of 5 days leaves 2 later days to check.
    Returning slices/views is fine: you will not modify these arrays.

    Independent API reminder — the stop index is excluded:
        values = np.array([10, 20, 30, 40])
        print(values[1:3])  # [20, 30]
    Slicing the first axis of a 2D array selects whole rows.

    Task: Return training sales, training actual, validation sales, and validation actual, in that order.
    Example input/output:
        sales = np.array([[0, 0], [1, 0], [0, 2], [1, 1], [2, 0]])
        actual = np.array([1., 3., 7., 6., 5.])
        train_days = 3
        Expected output: (np.array([[0, 0], [1, 0], [0, 2]]),
                          np.array([1., 3., 7.]),
                          np.array([[1, 1], [2, 0]]), np.array([6., 5.]))

    Pause: what goes wrong if sales and actual use different boundaries?
    """
    return sales[:train_days], actual[:train_days], sales[train_days:], actual[train_days:]


def predict_later_days(validation_sales: np.ndarray, weights: np.ndarray,
                       bias: float) -> np.ndarray:
    """2 — Use learned parameters without updating them (~5 minutes).

    Learn:
    A trained model can predict rows it has not used before. The formula
    remains sales @ weights + bias. The number of days may change: training
    could have 3 rows and validation 2, but the number and order of product
    columns must match the weights. Each new row gets one prediction.

    Prediction needs sales and parameters, not actual revenue. Validation
    targets are used later to check accuracy; they should not be needed
    just to produce a prediction. This function performs no update.

    Independent API reminder:
        a = np.array([2., 3.])
        b = np.array([4., 1.])
        print(a @ b)  # 11.0
    With a matrix on the left, @ repeats the dot product for every row.

    Task: Return revenue predictions for the supplied validation days.
    Example input/output:
        validation_sales = np.array([[1, 1], [2, 0]])
        weights = np.array([2., 3.])
        bias = 1.
        Expected output: np.array([6., 5.])

    Pause: why is there no actual argument or learning rate here?
    """
    pred = validation_sales @ weights + bias
    return pred


def validation_score(validation_sales: np.ndarray, validation_actual: np.ndarray,
                     weights: np.ndarray, bias: float) -> Tuple[np.ndarray, float]:
    """3 — Measure accuracy on the reserved days (~6 minutes).

    Learn:
    Predict with the supplied fixed model, subtract validation_actual,
    and average the squared errors. The result is validation MSE.
    It uses the same metric as training MSE, but on different observations.
    An error of -2 contributes 4, just like an error of +2.

    Do not calculate gradients or update parameters here. Evaluation
    asks how well this model predicts these days, not how well it could
    fit them after another round of training. Validation MSE can be higher
    OR lower than training MSE; there is no required ordering.

    Independent API reminders:
        print(np.array([-3., 1.]) ** 2)  # [9., 1.]
        print(float(np.mean(np.array([2., 6.]))))  # 4.0

    Task: Return validation predictions and their MSE without updating the model.
    Example input/output:
        validation_sales = np.array([[1, 1], [2, 0]])
        validation_actual = np.array([6., 5.])
        weights = np.array([1.3, 2.44])
        bias = 1.14
        Expected output: (np.array([4.88, 3.74]), 1.421)

    Pause: why must actual revenue stay separate from predicted revenue?
    """
    pred = validation_sales @ weights + bias
    signed_err = pred - validation_actual
    mse = np.mean(signed_err ** 2)
    return pred, float(mse)


def train_and_validate(train_sales: np.ndarray, train_actual: np.ndarray,
                       validation_sales: np.ndarray, validation_actual: np.ndarray,
                       weights: np.ndarray, bias: float,
                       learning_rate: float, steps: int
                       ) -> Tuple[np.ndarray, float, float, float]:
    """4 — Familiar training loop, then two separate measurements (~15 minutes).

    Learn:
    The arrays are already split. Train with train_sales and train_actual
    ONLY. In every update, error comes from the current TRAINING predictions.
    The formulas are unchanged: dw = (2/N_train) X_train^T error;
    db = 2*mean(error). N_train is the number of training rows, not the
    total number of rows across both sets. Both gradients use the old model.

    Each update subtracts learning_rate times the corresponding gradient.
    Carry the new weights and bias forward; refresh errors each iteration.
    Protect input weights with a copy and return independent weights even
    for zero steps. There is no history, early stopping, or rate search here.

    After exactly steps updates, freeze the parameters. Measure training
    MSE and validation MSE separately, both with this SAME final model.
    Never concatenate the validation rows into the training set or use
    validation errors to update parameters. If only validation targets
    change, your learned parameters and training MSE must stay identical.

    Independent API reminder:
        original = np.array([3., 5.])
        working = original.copy()
        working[0] = 9.
        print(original)  # [3., 5.]

    Task: Return learned weights, bias, final training MSE, and validation MSE.
    Example input/output:
        train_sales = np.array([[0, 0], [1, 0], [0, 2]])
        train_actual = np.array([1., 3., 7.])
        validation_sales = np.array([[1, 1], [2, 0]])
        validation_actual = np.array([6., 5.])
        weights = np.array([1., 1.])
        bias = 0.
        learning_rate = 0.15
        steps = 2
        Expected output: (np.array([1.3, 2.44]), 1.14, 0.4312, 1.421)

    Pause: which arrays should appear inside the gradient calculations?
    """
    weights = weights.copy()
    train_pred = train_sales @ weights + bias
    train_signed_err = train_pred - train_actual
    n_days = train_sales.shape[0]
    dw = (2 / n_days) * (train_sales.T @ train_signed_err)
    db = 2 * np.mean(train_signed_err)

    for _ in range(steps):
        weights = weights - (dw * learning_rate)
        bias -= (db * learning_rate)
        train_pred = train_sales @ weights + bias
        train_signed_err = train_pred - train_actual
        dw = (2 / n_days) * (train_sales.T @ train_signed_err)
        db = 2 * np.mean(train_signed_err)

    train_mse = np.mean(train_signed_err ** 2)

    validation_pred = validation_sales @ weights + bias
    validation_signed_err = validation_pred - validation_actual
    validation_mse = np.mean(validation_signed_err ** 2)

    return weights, float(bias), float(train_mse), float(validation_mse)



def difficult_validation_days(day_ids: np.ndarray, validation_sales: np.ndarray,
                              validation_actual: np.ndarray, weights: np.ndarray,
                              bias: float, threshold: float
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """5 — Short review: inspect errors instead of only their average (~6 minutes).

    Learn:
    A single MSE does not identify which dates need attention. Calculate
    signed errors for this fixed model, then select dates whose absolute
    error is strictly greater than threshold. threshold is nonnegative.
    Use the same boolean mask for IDs and errors to keep them aligned.

    Return SIGNED errors: negative means underprediction, positive means
    overprediction. Preserve input order; when none qualify, both returned
    arrays have shape (0,). As in exercise 3, do not update the model.
    The selected examples may guide investigation, but they do not become
    extra training rows in today's exercise.

    Independent API reminders:
        print(np.abs(np.array([-2., 3.])))  # [2., 3.]
        labels = np.array([8, 9, 10])
        print(labels[np.array([True, False, True])])  # [8, 10]

    Task: Return validation IDs with absolute error above threshold and their signed errors.
    Example input/output:
        day_ids = np.array([501, 502])
        validation_sales = np.array([[1, 1], [2, 0]])
        validation_actual = np.array([6., 5.])
        weights = np.array([1.3, 2.44])
        bias = 1.14
        threshold = 1.2
        Expected output: (np.array([502]), np.array([-1.26]))

    Pause: should a negative error be discarded before checking its magnitude?
    """
    validation_pred = validation_sales @ weights + bias
    signed_err = validation_pred - validation_actual
    keep = np.abs(signed_err) > threshold
    return day_ids[keep], signed_err[keep]


