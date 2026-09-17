"""Day 4 — How accurate is the shop's revenue prediction? (~40–50 min)

Rows of sales represent days; columns represent products. actual contains
OBSERVED daily revenue, which may differ from a model's predictions.
Weights and bias are given today. Training them comes later.

Work in order: predict → errors → MAE/MSE → filter → compare → report.
Write NumPy operations directly in each function, without calling earlier
exercise functions. Repeating prediction and error calculations is the goal.
No loops are needed; a Python if statement is fine for comparing models.

Assume valid, finite inputs: N,D >= 1; IDs are unique integers; sales are
nonnegative integers; weights, actual, and predictions are float arrays.
Weights and bias may be negative. Do not clip predictions or modify inputs.
actual and predicted always have shape (N,), NOT (N,1).

Every Task example lists its complete inputs and expected output.
Learn examples demonstrate individual operations, not complete solutions.
Replace only the empty implementations. From the repository root:
  python -m pytest -q week01_numpy_logreg/day04_evaluation -k predict_revenue
Replace the name after -k to test another function. Unfinished tests fail.
"""

from typing import Dict, Tuple, Union
import numpy as np


def predict_revenue(sales: np.ndarray, weights: np.ndarray,
                    bias: float) -> np.ndarray:
    """1 — Recall yesterday's linear model: X @ w + b.

    Learn: @ combines matching coordinates in a dot product.
    API example:
        a = np.array([2., 4.])
        b = np.array([3., 1.])
        print(a @ b)  # 10.0, a scalar
    For a matrix and vector: (N,D) @ (D,) -> (N,).

    Inputs: sales: int ndarray (N,D); weights: float ndarray (D,);
            bias: float.
    Output: float ndarray (N,).
    Task: Return the predicted daily revenue from the given linear model.
    Example input/output:
        sales = np.array([[2, 1], [0, 3], [1, 2]])
        weights = np.array([4., 2.])
        bias = 1.
        Expected output: np.array([11., 7., 9.])
    """
    return sales @ weights + bias



def prediction_errors(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """2 — Signed error: predicted minus actual.

    Learn: Positive error means overprediction; negative means
    underprediction. Array subtraction works element by element.
    API example:
        a = np.array([8., 3.])
        b = np.array([2., 5.])
        print(a - b)  # [6., -2.]

    Inputs: predicted, actual: float ndarrays, both (N,).
    Output: float ndarray (N,).
    Task: Return the signed prediction error for each day.
    Example input/output:
        predicted = np.array([11., 7., 9.])
        actual = np.array([10., 9., 6.])
        Expected output: np.array([1., -2., 3.])
    """
    return predicted - actual


def mean_absolute_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    """3 — MAE: the average magnitude of the errors.

    Learn: MAE = mean(|predicted - actual|). It has the same units as
    revenue. Taking magnitudes prevents positive/negative errors canceling.
    Independent API examples:
        print(np.abs(np.array([-4., 0., 7.])))  # [4., 0., 7.]
        print(np.mean(np.array([2., 5., 8.])))  # 5.0
    float(value) converts a NumPy scalar to a Python float.

    Inputs: predicted, actual: float ndarrays, both (N,).
    Output: Python float.
    Task: Return the mean absolute error across all days.
    Example input/output:
        predicted = np.array([11., 7., 9.])
        actual = np.array([10., 9., 6.])
        Expected output: 2.0
    """
    return float(np.mean(np.abs(predicted - actual)))


def mean_squared_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    """4 — MSE: the average squared error.

    Learn: MSE = mean((predicted - actual)^2). Larger errors receive
    more weight: doubling an error multiplies its square by four.
    MSE uses squared revenue units. In Python, square with **2, not ^2.
    API example:
        values = np.array([-3., 2., 0.])
        print(values ** 2)  # [9., 4., 0.]

    Inputs: predicted, actual: float ndarrays, both (N,).
    Output: Python float.
    Task: Return the mean squared error across all days.
    Example input/output:
        predicted = np.array([11., 7., 9.])
        actual = np.array([10., 9., 6.])
        Expected output: 4.666666666666667  # 14 / 3
    """
    return float(np.mean((predicted - actual) ** 2))


def inaccurate_days(day_ids: np.ndarray, sales: np.ndarray,
                    weights: np.ndarray, bias: float, actual: np.ndarray,
                    threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """5 — Repeat prediction, error calculation, and masking.

    Learn: Comparisons create masks; boolean indexing preserves order.
    Independent API examples:
        print(np.array([2., 5., 8.]) > 5.)  # [False, False, True]
        letters = np.array(['A', 'B', 'C'])
        print(letters[np.array([True, False, True])])  # ['A', 'C']

    Inputs: day_ids: int ndarray (N,); sales: int ndarray (N,D);
            weights: float ndarray (D,); bias: float;
            actual: float ndarray (N,); threshold: float >= 0.
    Output: tuple (int IDs (M,), float SIGNED errors (M,)), in input order;
            both arrays have shape (0,) when no day qualifies.
    Task: Return IDs and signed errors for days whose absolute error exceeds threshold.
    Example input/output:
        day_ids = np.array([101, 102, 103])
        sales = np.array([[2, 1], [0, 3], [1, 2]])
        weights = np.array([4., 2.])
        bias = 1.
        actual = np.array([10., 9., 6.])
        threshold = 1.
        Expected output: (np.array([102, 103]), np.array([-2., 3.]))
    """
    predicted = sales @ weights + bias
    abs_err = predicted - actual
    keep = np.abs(abs_err) > threshold
    return day_ids[keep], abs_err[keep]


def compare_models(sales: np.ndarray, actual: np.ndarray,
                   weights_a: np.ndarray, bias_a: float,
                   weights_b: np.ndarray, bias_b: float) -> Tuple[str, float, float]:
    """6 — Compare two fixed models on the SAME observations.

    Learn: Lower MSE means a closer fit on the data being evaluated.
    It does not by itself guarantee accuracy on future data.
    API example:
        print(float(np.mean(np.array([2., 4.]))))  # 3.0
    A tuple can hold different types, e.g. ('label', 1.0, 2.0).

    Inputs: sales: int ndarray (N,D); actual: float ndarray (N,);
            weights_a, weights_b: float ndarrays (D,);
            bias_a, bias_b: floats.
    Output: tuple (winner: str, mse_a: float, mse_b: float);
            winner is 'A' or 'B'; choose 'A' when the MSEs are equal.
    Task: Return the lower-MSE model's label and both models' MSE values.
    Example input/output:
        sales = np.array([[2, 1], [0, 3], [1, 2]])
        actual = np.array([10., 9., 6.])
        weights_a = np.array([4., 2.])
        bias_a = 1.
        weights_b = np.array([3., 3.])
        bias_b = 0.
        Expected output: ('B', 4.666666666666667, 3.3333333333333335)
    """
    predicted_a = sales @ weights_a + bias_a
    predicted_b = sales @ weights_b + bias_b
    mse_a = np.mean((predicted_a - actual) ** 2)
    mse_b = np.mean((predicted_b - actual) ** 2)
    return ('A' if mse_a <= mse_b else 'B'), float(mse_a), float(mse_b)


def evaluation_report(day_ids: np.ndarray, sales: np.ndarray,
                      weights: np.ndarray, bias: float,
                      actual: np.ndarray) -> Dict[str, Union[np.ndarray, float, int]]:
    """7 — Integrated repetition: evaluate one model.

    Learn: argmax returns a POSITION, not an ID. Ties use the first
    occurrence. Largest absolute error may be positive OR negative.
    API example:
        values = np.array([2., 8., 5.])
        print(np.argmax(values))  # 1
    int(value) converts a NumPy integer to a Python int.
    Dictionary syntax example: {'label': 'sample', 'count': 3}.

    Inputs: day_ids: int ndarray (N,); sales: int ndarray (N,D);
            weights: float ndarray (D,); bias: float;
            actual: float ndarray (N,).
    Output: dict with exactly these keys:
            'predictions': float ndarray (N,)
            'mse': Python float
            'worst_day_id': Python int (first in input order if tied)
    Task: Return predictions, MSE, and the ID of the day with largest absolute error.
    Example input/output:
        day_ids = np.array([101, 102, 103])
        sales = np.array([[2, 1], [0, 3], [1, 2]])
        weights = np.array([4., 2.])
        bias = 1.
        actual = np.array([10., 9., 6.])
        Expected output: {'predictions': np.array([11., 7., 9.]),
                          'mse': 4.666666666666667, 'worst_day_id': 103}
    """
    predicted = sales @ weights + bias
    mse = np.mean((predicted - actual) ** 2)
    abs_err = np.abs(predicted - actual)
    idx_max_abs_err = np.argmax(abs_err)
    return {
        'predictions': predicted,
        'mse': float(mse),
        'worst_day_id': int(day_ids[idx_max_abs_err])
    }
