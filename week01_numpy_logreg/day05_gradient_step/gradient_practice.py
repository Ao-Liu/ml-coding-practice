"""Day 5 — Adjust the shop's model once (~40–50 minutes).

1: recall prediction/errors/MSE; 2–3: bias; 4–5: weights;
6: one complete training step; 7: a short mask review (optional if tired).

Today's model is still prediction = X @ w + b. The difference: now we
calculate how to change w and b, rather than choosing them by hand.
Our loss is MSE = mean(error^2), WITHOUT a factor of 1/2.
Always use SIGNED error = prediction - actual when calculating gradients.

A gradient measures how loss changes when a parameter increases slightly.
To try to reduce loss, move AGAINST the gradient:
  new parameter = old parameter - learning_rate * gradient
A positive gradient asks for a decrease; a negative one asks for an increase.
The learning rate controls the size of that move. Too large can increase loss.

Formulas are provided: translate them into NumPy, rather than deriving them
from scratch today. Learn examples teach APIs only, not the task solutions.
Write operations directly in each function; don't call earlier exercises.
No loops are needed yet. Do not modify any input arrays.

Assume valid finite inputs, N,D >= 1, nonnegative integer sales, and positive
learning rates. Targets/weights may be any finite floats. Arrays use the
shapes specified in each function; actual is (N,), never (N,1).

Run one exercise from the repository root:
  python -m pytest -q week01_numpy_logreg/day05_gradient_step -k prediction_snapshot
Unfinished exercises intentionally fail with NotImplementedError.
"""

from typing import Tuple
import numpy as np


def prediction_snapshot(sales: np.ndarray, weights: np.ndarray,
                        bias: float, actual: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, float]:
    """1 — Warm-up: repeat yesterday's calculation.

    Learn: Prediction is a weighted sum plus bias; error is predicted
    minus actual; MSE is mean(error^2).
    Independent API examples:
        print(np.array([2., 3.]) @ np.array([4., 1.]))  # 11.0
        print(np.array([-2., 3.]) ** 2)               # [4., 9.]

    Inputs: sales: int ndarray (N,D); weights: float ndarray (D,);
            bias: float; actual: float ndarray (N,).
    Output: tuple (float predictions (N,), float signed errors (N,),
                   Python float MSE).
    Task: Return predictions, signed errors, and MSE for the given model.
    Example input/output:
        sales = np.array([[1, 0], [0, 1]])
        weights = np.array([0., 0.])
        bias = 0.
        actual = np.array([2., 4.])
        Expected output: (np.array([0., 0.]), np.array([-2., -4.]), 10.0)
    """
    predicted = sales @ weights + bias
    signed_errors = predicted - actual
    mse = np.mean(signed_errors ** 2)
    return predicted, signed_errors, float(mse)


def bias_gradient(sales: np.ndarray, weights: np.ndarray,
                  bias: float, actual: np.ndarray) -> float:
    """2 — First gradient: one shared bias.

    Learn: For MSE, db = 2 * mean(error). Increasing b raises every
    prediction equally; mean signed error tells us the overall direction.
    If predictions are mostly too low, db tends to be negative, so
    subtracting learning_rate * db raises b. The 2 comes from squaring.
    API example:
        values = np.array([-3., 1., 5.])
        print(np.mean(values))  # 1.0, a scalar
        print(float(np.sum(values)))  # 3.0, a Python float

    Inputs: sales: int ndarray (N,D); weights: float ndarray (D,);
            bias: float; actual: float ndarray (N,).
    Output: Python float db.
    Task: Return the MSE gradient with respect to bias at the given parameters.
    Example input/output:
        sales = np.array([[1, 0], [0, 1]])
        weights = np.array([0., 0.])
        bias = 0.
        actual = np.array([2., 4.])
        Expected output: -6.0
    """
    predicted = sales @ weights + bias
    errors = np.mean(predicted - actual)
    return float(2 * errors) # db 2 * mean(error) derivative


def update_bias(sales: np.ndarray, weights: np.ndarray, bias: float,
                actual: np.ndarray, learning_rate: float) -> Tuple[float, float]:
    """3 — Change bias once, then measure the NEW loss.

    Learn: Keep weights fixed here. Use db = 2 * mean(error), then
    b_new = b - learning_rate * db. The gradient uses the OLD model;
    the loss after the update needs NEW predictions.
    API example — scalar broadcasting:
        values = np.array([1., 4., 7.])
        print(values + 0.5)  # [1.5, 4.5, 7.5]

    Inputs: sales: int ndarray (N,D); weights: float ndarray (D,);
            bias: float; actual: float ndarray (N,); learning_rate: float > 0.
    Output: tuple (new_bias: Python float, mse_after: Python float).
    Task: Return the bias after one gradient step and the resulting MSE.
    Example input/output:
        sales = np.array([[1, 0], [0, 1]])
        weights = np.array([0., 0.])
        bias = 0.
        actual = np.array([2., 4.])
        learning_rate = 0.1
        Expected output: (0.6, 6.76)
    """
    predicted = sales @ weights + bias
    errors = np.mean(predicted - actual)
    db = 2 * errors
    b_new = bias - learning_rate * db
    predicted_new = sales @weights + b_new
    mse_new = np.mean((predicted_new - actual) ** 2)
    return float(b_new), float(mse_new)


def single_weight_gradient(units: np.ndarray, weight: float,
                           bias: float, actual: np.ndarray) -> float:
    """4 — One product: connect sales to its weight's gradient.

    Learn: With one product, prediction_i = units_i * weight + bias.
    The MSE gradient is dw = 2 * mean(units * error).
    Changing the weight changes a day's prediction in proportion to that
    day's units, which is why units multiply the signed error.
    API example — elementwise multiplication, not @:
        a = np.array([2., 5.])
        b = np.array([-1., 3.])
        print(a * b)  # [-2., 15.], shape (2,)

    Inputs: units: int ndarray (N,); weight, bias: floats;
            actual: float ndarray (N,).
    Output: Python float dw.
    Task: Return the MSE gradient for the single product's weight.
    Example input/output:
        units = np.array([1, 2])
        weight = 0.
        bias = 0.
        actual = np.array([2., 4.])
        Expected output: -10.0
    """
    predicted = units * weight + bias
    errors = predicted - actual
    return float(2 * np.mean(units * errors))



def weight_gradients(sales: np.ndarray, weights: np.ndarray,
                     bias: float, actual: np.ndarray) -> np.ndarray:
    """5 — Multiple products: transpose connects days to weights.

    Learn: dw_j = (2/N) * sum_i(sales_ij * error_i).
    In matrix notation: dw = (2/N) X^T error. X^T means transpose.
    Shapes: (D,N) @ (N,) -> (D,), one gradient per weight.
    API example — .T swaps the two axes of a 2D array:
        grid = np.array([[1, 2, 3], [4, 5, 6]])
        print(grid.T)        # [[1, 4], [2, 5], [3, 6]]
        print(grid.T.shape)  # (3, 2)
        print(grid.shape[0]) # 2: number of rows
    Unlike reshape, transpose swaps which values belong to each row.

    Inputs: sales: int ndarray (N,D); weights: float ndarray (D,);
            bias: float; actual: float ndarray (N,).
    Output: float ndarray (D,) containing dw.
    Task: Return all weight gradients using transpose and matrix multiplication.
    Example input/output:
        sales = np.array([[1, 0], [0, 1]])
        weights = np.array([0., 0.])
        bias = 0.
        actual = np.array([2., 4.])
        Expected output: np.array([-2., -4.])
    """
    predicted = sales @ weights + bias
    errors = predicted - actual
    num_days = np.shape(sales)[0]
    dw = (2 / num_days) * (sales.T @ errors) # T是row -> col
    return dw


def train_step(sales: np.ndarray, weights: np.ndarray, bias: float,
               actual: np.ndarray, learning_rate: float
               ) -> Tuple[np.ndarray, float, float, float]:
    """6 — One complete training step (no training loop yet).

    Learn: Use dw = (2/N) X^T error and db = 2 * mean(error).
    BOTH gradients must use the SAME old predictions. Update each
    parameter by subtracting learning_rate * its gradient, then evaluate
    the updated model. A large learning rate may increase MSE.
    Independent API examples:
        values = np.array([3., 5.])
        shifted = values - 1.
        print(values)   # [3., 5.]: the input was not changed
        print(shifted)  # [2., 4.]: a separate result

    Inputs: sales: int ndarray (N,D); weights: float ndarray (D,);
            bias: float; actual: float ndarray (N,); learning_rate: float > 0.
    Output: tuple (new_weights: float ndarray (D,), new_bias: Python float,
                   mse_before: Python float, mse_after: Python float).
    Task: Return updated weights and bias plus MSE before and after one step.
    Example input/output:
        sales = np.array([[1, 0], [0, 1]])
        weights = np.array([0., 0.])
        bias = 0.
        actual = np.array([2., 4.])
        learning_rate = 0.1
        Expected output: (np.array([0.2, 0.4]), 0.6, 10.0, 5.22)
    Experiment after passing: for these SAME inputs except learning_rate=1.,
        expected output is (np.array([2., 4.]), 6.0, 10.0, 36.0).
        The larger step makes the loss worse. Explain why before Day 6.
    """
    predicted = sales @ weights + bias
    errors = predicted - actual
    num_days = np.shape(sales)[0]
    dw = (2 / num_days) * (sales.T @ errors)
    db = 2 * np.mean(errors)

    new_w = weights - dw * learning_rate # 减去 因为梯度指向 loss 增大的方向，而我们想让 loss 变小，所以要反着走。
    new_b = bias - db * learning_rate
    new_p = sales @ new_w + new_b
    return new_w, float(new_b), float(np.mean(errors ** 2)), float(np.mean((new_p - actual) ** 2))


def remaining_bad_days(day_ids: np.ndarray, sales: np.ndarray,
                       updated_weights: np.ndarray, updated_bias: float,
                       actual: np.ndarray, threshold: float
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """7 — Short review: mask errors after an update (optional).

    Learn: Select by absolute error but return SIGNED errors, as on Day 4.
    The provided parameters are already updated; do not update them again.
    Independent API examples:
        print(np.abs(np.array([-3., 1.])))  # [3., 1.]
        labels = np.array([8, 9, 10])
        print(labels[np.array([True, False, True])])  # [8, 10]

    Inputs: day_ids: int ndarray (N,); sales: int ndarray (N,D);
            updated_weights: float ndarray (D,); updated_bias: float;
            actual: float ndarray (N,); threshold: float >= 0.
    Output: tuple (int IDs (M,), float signed errors (M,)), in input order;
            both shapes are (0,) when no day qualifies.
    Task: Return IDs and signed errors whose absolute error exceeds threshold.
    Example input/output:
        day_ids = np.array([101, 102])
        sales = np.array([[1, 0], [0, 1]])
        updated_weights = np.array([0.2, 0.4])
        updated_bias = 0.6
        actual = np.array([2., 4.])
        threshold = 2.
        Expected output: (np.array([102]), np.array([-3.]))
    """
    predicted = sales @ updated_weights + updated_bias
    errors = predicted - actual
    abs_errs = np.abs(errors)
    keep = abs_errs > threshold
    return day_ids[keep], errors[keep]
