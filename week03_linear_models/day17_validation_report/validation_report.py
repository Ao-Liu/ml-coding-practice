"""Day 17 — Make the baseline/model comparison visible (~35–40 minutes).

The shop already has fitted weights and saved training statistics. Today focus
on evaluating them, with no new gradient rule or training loop. The only new
NumPy convenience is np.full, which repeats a constant into an array.
Use direct operations in exercises 1–4; exercise 5 may reuse your functions.
All arrays are finite, all datasets nonempty, all supplied scales positive.
Do not modify inputs. Day IDs are integers; numeric results are floats.
"""
from typing import Tuple
import numpy as np


def baseline_predictions(train_actual: np.ndarray, n_days: int) -> np.ndarray:
    """1 — Write down the baseline's prediction for every day (~6 minutes).

    Learn:
    The baseline always guesses the mean TRAINING revenue. It is separate from
    the model's first prediction and never uses sales, weights, or bias.
    If training revenues are [2,6], its guess is 4 for every future day.
    Today make that repeated guess visible as an array with shape (n_days,).
    n_days is the number of days to predict, not the number of training rows.
    It is positive. Compute one mean, then fill the output with that value.
    np.full(shape, fill_value) builds an array; for a 1D array shape can be an
    integer. Use dtype=float to request floating-point entries explicitly.

    Independent API example:
        placeholders = np.full(4, -2., dtype=float)
        print(placeholders)        # [-2.,-2.,-2.,-2.]
        print(placeholders.shape)  # (4,)

    Task: Return a float array of n_days copies of the training mean revenue.
    Example input/output:
        train_actual = np.array([1.,5.])
        n_days = 3
        Expected output: np.array([3.,3.,3.])

    Pause: would changing validation targets change this baseline prediction?
    """
    mean = np.mean(train_actual)
    return np.full(n_days, mean, dtype=float)


def revenue_predictions(sales: np.ndarray, means: np.ndarray,
                        safe_scales: np.ndarray, weights: np.ndarray,
                        bias: float) -> np.ndarray:
    """2 — Review: standardized features are not revenue predictions (~7 minutes).

    Learn:
    Subtract supplied training means and divide by supplied safe scales.
    That produces standardized FEATURES, still a table (N,D). Then multiply
    that table by fitted weights (D,) and add bias to get revenues (N,).
    These are two different steps with different meanings and shapes.
    Do not subtract actual revenues from the (N,D) feature table: even when
    broadcasting permits it, those numbers do not represent prediction errors.
    Keep training statistics fixed and do not update model parameters here.
    Safe scales already include any replacements for zero training std.

    Independent API reminder:
        grid = np.array([[2.,0.], [1.,3.], [0.,4.]])
        print(grid @ np.array([4.,2.]))  # [8.,10.,8.], shape (3,)

    Task: Return one predicted revenue per raw sales row using the supplied model and statistics.
    Example input/output:
        sales = np.array([[2,7], [4,7], [1,7]])
        means = np.array([2.,7.])
        safe_scales = np.array([1.,1.])
        weights = np.array([2.,0.])
        bias = 3.
        Expected output: np.array([3.,7.,1.])

    Pause: for three days and two products, what shape exists before and after @?
    """
    standardized = (sales - means) / safe_scales
    return standardized @ weights + bias



def daily_squared_errors(actual: np.ndarray, baseline_pred: np.ndarray,
                         model_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """3 — Compare both predictors against the same answers (~6 minutes).

    Learn:
    actual, baseline_pred, and model_pred all have shape (N,) in the same day
    order. For EACH predictor separately, subtract actual and square the result.
    Keep one squared error per day; do not take a mean yet.
    For a true revenue 6, guesses 3 and 7 have signed errors -3 and +1 and
    squared errors 9 and 1. The second prediction is closer on that day.
    Comparing the predictions to each other would measure disagreement, not
    accuracy. The actual revenues are the answers that score both predictors.

    Independent API reminder:
        values = np.array([-2.,0.,3.])
        print(values ** 2)  # [4.,0.,9.]

    Task: Return baseline squared errors followed by model squared errors, both as arrays.
    Example input/output:
        actual = np.array([4.,6.,2.])
        baseline_pred = np.array([3.,3.,3.])
        model_pred = np.array([3.,7.,1.])
        Expected output: (np.array([1.,9.,1.]), np.array([1.,1.,1.]))

    Pause: can equally large positive and negative signed errors have different squared errors?
    """
    baseline_signed_err = baseline_pred - actual
    model_signed_err = model_pred - actual
    return baseline_signed_err ** 2, model_signed_err ** 2


def comparison_scores(baseline_squared: np.ndarray, model_squared: np.ndarray
                      ) -> Tuple[float, float, float]:
    """4 — Two scores and one difference (~6 minutes).

    Learn:
    These arrays ALREADY contain squared errors for the same days. Average
    each separately to get its MSE. Do not square these numbers again.
    The third result is baseline MSE minus model MSE, an absolute reduction
    in error. It is calculated from the first two scores, not a third model.
    If baseline MSE=10 and model MSE=6, reduction=4 means the model is better.
    If model MSE=12 instead, reduction=-2 means worse. Keep that negative sign.
    This is not a percentage or a guarantee that training always helps.

    Independent API reminder:
        values = np.array([2.,4.,9.])
        print(float(values.mean()))  # 5.0

    Task: Return baseline MSE, model MSE, and baseline MSE minus model MSE as Python floats.
    Example input/output:
        baseline_squared = np.array([1.,9.,1.])
        model_squared = np.array([1.,1.,1.])
        Expected output: (11/3, 1.0, 8/3)

    Pause: if both predictors have the same MSE, what is the third number?
    """
    baseline_mse = np.mean(baseline_squared)
    model_mse = np.mean(model_squared)
    return float(baseline_mse), float(model_mse), float(baseline_mse - model_mse)


def evaluate_days(day_ids: np.ndarray, validation_sales: np.ndarray,
                  validation_actual: np.ndarray, train_actual: np.ndarray,
                  means: np.ndarray, safe_scales: np.ndarray,
                  weights: np.ndarray, bias: float
                  ) -> Tuple[float, float, float, np.ndarray]:
    """5 — Assemble a small validation report (~12 minutes).

    Learn:
    Start with two predictions for every validation day: a constant chosen
    from train_actual, and a fitted model prediction from standardized sales.
    Score both against validation_actual, keeping daily squared errors first.
    Average those arrays to get the two MSEs, then subtract model MSE from
    baseline MSE. The supplied model and training statistics stay fixed.
    You may reuse exercises 1–4; the purpose is to connect their meanings.

    Also return IDs of days where model squared error is STRICTLY smaller
    than baseline squared error. Compare the daily arrays to make a mask,
    then select day_ids with it. Preserve order; ties do not count. When
    nothing improves, return an integer array of shape (0,), not a list.
    Winning on more days does not guarantee lower overall MSE: one very large
    error can outweigh several small wins. The scores and IDs answer different
    questions: how much error overall, and which individual days improved?
    Return (baseline MSE, model MSE, MSE reduction, improved IDs), in that order.
    The three scores are Python floats. No training or new statistics here.

    Independent API reminder:
        ids = np.array([10,20,30])
        keep = np.array([False,True,False])
        print(ids[keep])  # [20]

    Task: Return both validation MSEs, their baseline-minus-model difference, and strictly improved day IDs.
    Example input/output:
        day_ids = np.array([101,102,103])
        validation_sales = np.array([[2,7], [4,7], [1,7]])
        validation_actual = np.array([4.,6.,2.])
        train_actual = np.array([1.,5.])
        means = np.array([2.,7.])
        safe_scales = np.array([1.,1.])
        weights = np.array([2.,0.])
        bias = 3.
        Expected output: (11/3, 1.0, 8/3, np.array([102]))

    Pause: if validation_actual changes, should either set of predictions change?
    """
    baseline_mean = np.mean(train_actual)
    n_days = day_ids.shape[0]
    baseline_pred = np.full(n_days, baseline_mean, dtype=float)
    baseline_signed_err =  baseline_pred - validation_actual
    baseline_mse = np.mean(baseline_signed_err ** 2)

    standardized = (validation_sales - means) / safe_scales
    model_pred = standardized @ weights + bias
    model_signed_err = model_pred - validation_actual
    model_mse = np.mean(model_signed_err ** 2)

    mask = (model_signed_err ** 2) < (baseline_signed_err ** 2)

    return float(baseline_mse), float(model_mse), float(baseline_mse - model_mse), day_ids[mask]

