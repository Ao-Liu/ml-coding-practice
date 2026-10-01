"""Day 18 — Choose among fitted models on the same validation days (~40 minutes).

One small new idea: choose the candidate with lowest validation MSE, using
np.argmin to find its index. No new model or gradient formula today.
In candidate tables, ROWS ARE MODELS and COLUMNS ARE DAYS, not product features.
Each candidate was fitted without validation data; predictions are already given
in exercises 2–5. Validation may choose a model without updating its weights.
After selection, an untouched test set would be needed for a final independent
assessment; today's selected validation score is not that final assessment.
All arrays are finite, K>=1 candidates, N>=1 days, and inputs stay unchanged.
Repeat operations directly in 1–4; exercise 5 may reuse your earlier functions.
"""
from typing import Tuple
import numpy as np


def predict_candidate(sales: np.ndarray, means: np.ndarray, safe_scales: np.ndarray,
                      weights: np.ndarray, bias: float) -> np.ndarray:
    """1 — Review where one candidate's predictions come from (~6 minutes).

    Learn:
    This is one already fitted model. Transform raw sales with its supplied
    training means and positive safe scales, then apply its weights and bias.
    Standardized features are (N,D); predictions are (N,). Keep both stages
    distinct. No target values are needed to make these predictions.
    A different fitted model may predict different values on the same days.
    In later exercises those prediction arrays are arranged as rows of a table.
    Each model must be evaluated on the same days in the same order.

    Independent API reminder:
        grid = np.array([[2.,0.], [1.,3.], [0.,4.]])
        print(grid @ np.array([4.,2.]))  # [8.,10.,8.]

    Task: Return one candidate model's revenue predictions for the supplied raw sales.
    Example input/output:
        sales = np.array([[2,7], [4,7], [1,7]])
        means = np.array([2.,7.])
        safe_scales = np.array([1.,1.])
        weights = np.array([2.,0.])
        bias = 3.
        Expected output: np.array([3.,7.,1.])

    Pause: which shape describes features, and which describes predicted revenue?
    """
    standardized = (sales - means) / safe_scales
    prediction = standardized @ weights + bias
    return prediction


def candidate_mses(predictions: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """2 — Review broadcasting and axis: one MSE per model (~8 minutes).

    Learn:
    Here predictions has shape (K,N): K MODELS, each predicting the same N DAYS.
    This is a prediction table, not sales. actual has shape (N,). Subtracting
    actual broadcasts the same daily answers across every candidate row.
    Squaring gives (K,N) daily squared errors. Average ACROSS DAYS within each
    row (axis=1), leaving (K,), one MSE per model. Do not average all cells into
    one score or average across models: we need to compare candidates separately.
    For actual [2,4], a candidate row [1,6] has errors [-1,2], squares [1,4],
    and MSE 2.5. Repeat that meaning for every row without a Python loop.

    Independent API reminder:
        grid = np.array([[2.,4.,6.], [1.,3.,8.]])
        print(grid - np.array([1.,2.,3.]))  # [[1.,2.,3.], [0.,1.,5.]]
        print(grid.mean(axis=1))  # [4.,4.], one mean per row

    Task: Return a float array containing each candidate's validation MSE in row order.
    Example input/output:
        predictions = np.array([[3.,3.,3.], [3.,7.,1.]])
        actual = np.array([4.,6.,2.])
        Expected output: np.array([11/3, 1.0])

    Pause: why is axis=1 correct here even though sales column means used axis=0?
    """
    signed_err = predictions - actual
    return np.mean(signed_err ** 2, axis=1)


def best_candidate(mses: np.ndarray) -> int:
    """3 — New: select the position of the smallest score (~6 minutes).

    Learn:
    Lower MSE is better. np.min returns the smallest VALUE; np.argmin returns
    the INDEX where it occurs. We need the index so we can find that model's
    prediction row later. Indices start at zero. Return a Python int.
    If multiple candidates tie, np.argmin chooses the first occurrence; use
    that same rule today. Do not sort the scores, which would lose row identity.
    The winner is best among these candidates on this validation set. It need
    not beat a baseline and is not guaranteed to be best on future data.

    Independent API example:
        values = np.array([8.,2.,5.,2.])
        print(np.min(values))          # 2.0: the value
        print(int(np.argmin(values)))  # 1: the first position holding that value

    Task: Return the index of the lowest MSE, choosing the first index in a tie.
    Example input/output:
        mses = np.array([11/3, 1.0, 2.0])
        Expected output: 1

    Pause: could the minimum MSE be 0.5 while its index is 2?
    """
    return int(np.argmin(mses))


def improved_ids(day_ids: np.ndarray, actual: np.ndarray,
                 selected_predictions: np.ndarray, train_actual: np.ndarray) -> np.ndarray:
    """4 — Review daily masks for the selected model (~7 minutes).

    Learn:
    The baseline guesses mean(train_actual), not mean sales or validation
    targets. Compare its squared error with selected_predictions' squared error
    on each validation day. Both error arrays have shape (N,).
    Keep day_ids only where the model error is strictly smaller. Equal errors
    do not count. Preserve input order; no matches means an integer array (0,).
    Do NOT compare the two overall MSEs to build this mask: that gives one
    boolean instead of a separate decision for every day.

    Independent API reminder:
        ids = np.array([8,3,5])
        errors = np.array([2.,7.,1.])
        print(ids[errors < 3.])  # [8,5]

    Task: Return IDs where the selected model beats the training-mean baseline in squared error.
    Example input/output:
        day_ids = np.array([101,102,103])
        actual = np.array([4.,6.,2.])
        selected_predictions = np.array([3.,7.,1.])
        train_actual = np.array([1.,5.])
        Expected output: np.array([102])

    Pause: why can some days tie even when the model's overall MSE is better?
    """
    baseline_mean = np.mean(train_actual)
    baseline_predictions = np.full(day_ids.shape[0], baseline_mean, dtype=float)
    baseline_signed_err = baseline_predictions - actual

    model_signed_err = selected_predictions - actual
    mask = (model_signed_err ** 2) < (baseline_signed_err ** 2)
    return day_ids[mask]



def selection_report(day_ids: np.ndarray, predictions: np.ndarray,
                     validation_actual: np.ndarray, train_actual: np.ndarray
                     ) -> Tuple[int, float, float, np.ndarray]:
    """5 — Connect selection and baseline comparison (~12 minutes).

    Learn:
    predictions is (K,N), with one fitted model per row and one validation day
    per column. Compute one MSE per row using the same validation_actual (N,).
    Select the first minimum's index, then select that candidate's entire
    prediction row for the daily comparison. You may reuse exercises 2–4.
    Indexing a (K,N) table with one integer selects a row of shape (N,).

    Separately compute baseline validation MSE using mean(train_actual) as the
    constant prediction. Return the selected index, its MSE, the improvement
    baseline MSE minus selected MSE, and its strictly improved day IDs.
    The index is a Python int; both scores are Python floats. Do not clip a
    negative improvement: the best candidate may still lose to the baseline.
    Preserve ID order, use first-index tie breaking, and leave inputs unchanged.
    Nothing is trained here. Validation chooses between fixed candidates; it
    does not recompute their preprocessing statistics or change their weights.

    Independent API reminder:
        table = np.array([[2.,5.,8.], [1.,4.,7.]])
        row_index = 1
        print(table[row_index])        # [1.,4.,7.]
        print(table[row_index].shape)  # (3,)

    Task: Return the selected index, its MSE, its baseline-minus-model improvement, and improved IDs.
    Example input/output:
        day_ids = np.array([101,102,103])
        predictions = np.array([[3.,3.,3.], [3.,7.,1.], [4.,8.,0.]])
        validation_actual = np.array([4.,6.,2.])
        train_actual = np.array([1.,5.])
        Expected output: (1, 1.0, 8/3, np.array([102]))

    Pause: does choosing the best of three models guarantee it beats the baseline?
    """
    baseline_mean = np.mean(train_actual)
    baseline_predictions = np.full(day_ids.shape[0], baseline_mean, dtype=float)
    baseline_mse = np.mean((baseline_predictions - validation_actual) ** 2)

    signed_err = predictions - validation_actual
    mses = np.mean(signed_err ** 2, axis=1)
    idx = np.argmin(mses)
    model_mse = mses[idx]

    return int(idx), float(model_mse), baseline_mse - model_mse, day_ids[
        (baseline_predictions - validation_actual) ** 2 > (predictions[idx] - validation_actual) ** 2
    ]



