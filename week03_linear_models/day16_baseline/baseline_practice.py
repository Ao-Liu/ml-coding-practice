"""Day 16 — Does the sales model beat a simple guess? (~40 minutes).

One new idea: a baseline predicts the TRAINING mean revenue for every day.
Evaluate baseline and model on the SAME validation days. A lower training loss
alone does not answer which predicts new days better. No new gradient formula.
Arrays are finite, datasets nonempty, and input arrays must stay unchanged.
Repeat NumPy directly in exercises 1–4; exercise 5 may reuse earlier helpers.
"""
from typing import Tuple
import numpy as np


def baseline_revenue(train_actual: np.ndarray) -> float:
    """1 — New idea, familiar mean: a constant prediction (~5 minutes).

    Learn:
    Before building a complicated predictor, make a simple reference to beat.
    Our baseline ignores sales and predicts one number for every future day:
    the average TRAINING revenue. For revenues [2,6], it always predicts 4.
    This does not mean every day really earns 4; it is a deliberately simple
    guess. The training mean minimizes training squared error among constant
    predictions. It is not guaranteed to minimize validation error.
    Only training targets choose the guess. Looking at validation targets to
    choose it would give the baseline information unavailable at prediction time.

    Independent API reminder:
        values = np.array([3., 9., 12.])
        print(float(np.mean(values)))  # 8.0

    Task: Return the training mean revenue as a Python float.
    Example input/output:
        train_actual = np.array([1.,3.,8.])
        Expected output: 4.0

    Pause: does this baseline use any product's sales?
    """
    return float(np.mean(train_actual))


def baseline_mse(train_actual: np.ndarray, validation_actual: np.ndarray) -> float:
    """2 — Review scalar broadcasting and MSE (~6 minutes).

    Learn:
    Choose the constant using training revenues, then compare that SAME number
    with every validation target. Scalar subtraction broadcasts across the
    validation array: no loop or repeated prediction array is necessary.
    Square the signed errors and average over validation days. Training and
    validation may have different lengths. Do not subtract targets pairwise.
    With a constant guess 4 and actual revenues [3,7], errors are [1,-3];
    their squares are [1,9], so MSE is 5. This is a validation measurement.

    Independent API reminder:
        values = np.array([2., 5., 9.])
        print(6. - values)  # [4.,1.,-3.]
        print(values ** 2)  # [4.,25.,81.]

    Task: Return validation MSE for the constant training-mean predictor.
    Example input/output:
        train_actual = np.array([1.,3.,8.])
        validation_actual = np.array([3.,7.])
        Expected output: 5.0

    Pause: which array chooses the prediction, and which measures its error?
    """
    train_mean = float(np.mean(train_actual))
    signed_err = validation_actual - train_mean
    return float(np.mean(signed_err ** 2))

def model_mse(sales: np.ndarray, actual: np.ndarray, means: np.ndarray,
              safe_scales: np.ndarray, weights: np.ndarray, bias: float) -> float:
    """3 — Review the fitted model's evaluation (~7 minutes).

    Learn:
    This model learned weights on standardized sales. Transform raw sales with
    the supplied TRAINING means and safe scales, then predict and calculate MSE.
    All supplied scales are positive; replacement of zero std has already
    happened. Do not recompute statistics or update parameters here.
    The transformed table is (N,D), weights are (D,), and predictions are (N,).
    actual has the same N days, still in revenue units. Both this model and the
    baseline are scored against the same actual values for a fair comparison.

    Independent API reminder:
        grid = np.array([[2.,1.], [0.,3.]])
        print(grid @ np.array([4.,2.]))  # [10.,6.]

    Task: Return MSE for the supplied fitted model on raw sales and actual revenues.
    Example input/output:
        sales = np.array([[2,7], [4,7]])
        actual = np.array([3.,7.])
        means = np.array([2.,7.])
        safe_scales = np.array([1.,1.])
        weights = np.array([1.5,0.])
        bias = 3.
        Expected output: 0.5

    Pause: why would using raw sales directly with these weights be wrong?
    """
    standardized = (sales - means) / safe_scales
    pred = standardized @ weights + bias
    signed_err = pred - actual
    return float(np.mean(signed_err ** 2))


def improved_days(day_ids: np.ndarray, actual: np.ndarray,
                  model_predictions: np.ndarray, baseline_prediction: float) -> np.ndarray:
    """4 — Review masks: where did the model help? (~7 minutes).

    Learn:
    Compare squared errors separately for each day, before taking any mean.
    Keep an ID only when the model's squared error is STRICTLY smaller than
    the baseline's squared error. Equal errors do not count as an improvement.
    The mask has shape (N,) and selects matching IDs in their original order.
    Return an array of IDs, including shape (0,) when none qualify.
    A model can improve some days and worsen others; this mask alone does not
    establish that its overall MSE improved. Error magnitudes also matter.
    This function receives predictions already calculated, so do not transform
    them or retrain. Both predictors are judged against the same actual values.

    Independent API reminder:
        labels = np.array([11,22,33])
        scores = np.array([4.,1.,4.])
        print(labels[scores < 4.])  # [22]

    Task: Return IDs of days whose model squared error is smaller than baseline squared error.
    Example input/output:
        day_ids = np.array([101,102,103])
        actual = np.array([3.,7.,4.])
        model_predictions = np.array([3.,6.,5.])
        baseline_prediction = 4.
        Expected output: np.array([101,102])

    Pause: if one prediction misses by +2 and another by -2, is either better here?
    """
    baseline_signed_err = baseline_prediction - actual
    signed_err = model_predictions - actual
    mask = signed_err **2 < baseline_signed_err ** 2
    return day_ids[mask]


def train_and_compare(train_sales: np.ndarray, train_actual: np.ndarray,
                      validation_sales: np.ndarray, validation_actual: np.ndarray,
                      weights: np.ndarray, bias: float, learning_rate: float, steps: int
                      ) -> Tuple[float, float, float]:
    """5 — Repeat training, then compare with the baseline (~15 minutes).

    Learn:
    Fit the familiar model using only training data. Compute column means and
    population std (axis=0, ddof=0), replace exactly zero scales with 1, and
    standardize training sales. Copy weights so updates leave inputs unchanged.
    Repeat exactly steps updates; zero steps evaluates the initial parameters.
    Recall errors = X @ weights + bias - train_actual, where X is standardized.
    dw = (2/N) * (X.T @ errors), db = 2 * mean(errors), and N counts training
    days. Both gradients use the same current errors. Subtract learning_rate
    times each gradient and recompute errors for the next iteration.

    After fitting, use the saved training statistics to transform validation
    sales and evaluate the fitted model. Separately evaluate the constant
    training-mean baseline on those same validation targets. Never standardize
    revenue targets or use validation information for fitting either predictor.
    You may reuse exercises 1–3 for evaluation; repeat the training loop here.

    Return baseline MSE, model MSE, and improvement = baseline MSE - model MSE.
    A positive improvement means the model did better; negative means worse;
    zero means a tie. Do not clip negative results. Improvement is an absolute
    MSE difference, not a percentage, and no improvement is guaranteed.

    Independent API reminder:
        first_score, second_score = (9.0, 6.5)
        print(first_score - second_score)  # 2.5

    Task: Return validation baseline MSE, model MSE, and their improvement as three Python floats.
    Example input/output:
        train_sales = np.array([[1,7], [3,7]])
        train_actual = np.array([1.,5.])
        validation_sales = np.array([[2,7], [4,7]])
        validation_actual = np.array([3.,7.])
        weights = np.array([0.,0.])
        bias = 0.
        learning_rate = 0.1
        steps = 2
        Expected output: (8.0, 11.8784, -3.8784)

    Pause: this model trained for two steps but still loses to the baseline;
    does a decreasing training loss guarantee it already beats the baseline?
    """
    train_means = train_sales.mean(axis=0)
    train_std = train_sales.std(axis=0, ddof=0)
    train_std[train_std == 0] = 1
    train_standardized = (train_sales - train_means) / train_std
    predictions = train_standardized @ weights + bias
    signed_err = predictions - train_actual
    n = train_standardized.shape[0]

    for _ in range(steps):
        dw = (2 / n) * (train_standardized.T @ signed_err)
        db = 2 * np.mean(signed_err)
        weights = weights - (learning_rate * dw)
        bias -= (learning_rate * db)
        predictions = train_standardized @ weights + bias
        signed_err = predictions - train_actual

    validation_standardized = (validation_sales - train_means) / train_std # this calculates standardized
    validation_predictions = validation_standardized @ weights + bias
    validation_sign_err = validation_predictions - validation_actual
    m_mse = np.mean(validation_sign_err ** 2)

    baseline_means = np.mean(train_actual) #use actual
    baseline_signed_err = baseline_means - validation_actual
    b_mse = np.mean(baseline_signed_err ** 2)

    return float(b_mse), float(m_mse), float(b_mse - m_mse)



