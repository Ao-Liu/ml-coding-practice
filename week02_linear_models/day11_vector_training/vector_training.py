"""Day 11 — Train the vector model, one familiar step at a time.

Five exercises, about 35–45 minutes. Same shop and MSE as yesterday.
sales is (N,D): rows are days, columns are products. weights is (D,),
actual is (N,), and bias is one float. One, two, or three products should
work with the same code; no separate weight_a/weight_b variables are needed.

Recall: predictions = X @ w + b; error = predictions - actual;
        dw = (2/N) X^T error; db = 2*mean(error).
N is the number of DAYS, not the number of products. MSE = mean(error^2).

Assume valid finite inputs and intermediate values, N,D >= 1, nonnegative
integer sales, float weights/actual, positive learning_rate, and integer
steps >= 0. day_ids are unique integers; threshold >= 0. Do not modify inputs.
Return floating arrays and Python floats, except selected IDs retain integer
values. Write operations directly in each function instead of calling earlier
exercises. Use loops over training steps only from exercise 3 onward.

From the repository root:
  python -m pytest -q week02_linear_models/day11_vector_training -k one_step
Unfinished functions intentionally raise NotImplementedError.
"""

from typing import Tuple
import numpy as np


def one_step(sales: np.ndarray, weights: np.ndarray, bias: float,
             actual: np.ndarray, learning_rate: float
             ) -> Tuple[np.ndarray, float, float]:
    """1 — Recall yesterday's vector update (~5 minutes).

    Learn:
    Predict once with the old model, then use those signed errors in BOTH
    gradients: dw = (2/N) X^T error and db = 2*mean(error). Subtract the
    learning rate times each gradient from its corresponding parameter.
    Evaluate the updated model with fresh predictions to obtain its MSE.

    Think about the shapes before coding: the old errors are (N,), the
    transpose is (D,N), and their @ result is (D,), one number per weight.
    Elementwise * alone leaves a table of contributions rather than
    summing across days. Dividing by N converts those sums into averages.

    Independent API reminder:
        a = np.array([2., 4.])
        b = np.array([3., 1.])
        print(a * b)  # [6., 4.]
        print(a @ b)  # 10.0

    Task: Return new weights, new bias, and MSE after one joint update.
    Example input/output:
        sales = np.array([[0, 0], [1, 0], [0, 2]])
        weights = np.array([1., 1.])
        bias = 0.
        actual = np.array([1., 3., 7.])
        learning_rate = 0.15
        Expected output: (np.array([1.2, 2.]), 0.8, 1.96)

    Pause: what would change if you accidentally divided by weights.size?
    """
    pred = sales @ weights + bias
    signed_err = pred - actual
    n_days = sales.shape[0]
    dw = (2 / n_days) * (sales.T @ signed_err)
    db = 2 * np.mean(signed_err)

    weights = weights - (dw * learning_rate)
    bias -= (db * learning_rate)
    pred = sales @ weights + bias
    signed_err = pred - actual

    return weights, float(bias), float(np.mean(signed_err ** 2))



def two_steps(sales: np.ndarray, weights: np.ndarray, bias: float,
              actual: np.ndarray, learning_rate: float
              ) -> Tuple[np.ndarray, float, float]:
    """2 — Two updates, without a loop yet (~7 minutes).

    Learn:
    After the first update, both the vector of weights and the bias have
    changed. The second step starts from this new pair. Calculate new
    predictions, errors, and BOTH gradients before making the second move.
    Reusing the first dw/db would follow the slope of the old model.

    Write two blocks today, just as you did for bias on Day 7. The data,
    N, and learning_rate stay fixed; model parameters and all values derived
    from them change. Within one step, both gradients must use one model
    state. Between steps, refresh that state.

    Independent Python reminder:
        value = 2.
        saved = value * 3
        value = 5.
        print(saved)  # 6.0: expressions are not recalculated automatically

    Task: Return weights, bias, and MSE after exactly two joint updates.
    Example input/output:
        sales = np.array([[0, 0], [1, 0], [0, 2]])
        weights = np.array([1., 1.])
        bias = 0.
        actual = np.array([1., 3., 7.])
        learning_rate = 0.15
        Expected output: (np.array([1.3, 2.44]), 1.14, 0.4312)

    Pause: which model supplies the errors for the second weight gradient?
    """
    pred = sales @ weights + bias
    signed_err = pred - actual
    n_days = sales.shape[0]
    dw = (2 / n_days) * (sales.T @ signed_err)
    db = 2 * np.mean(signed_err)

    weights = weights - (dw * learning_rate)
    bias -= (db * learning_rate)
    pred = sales @ weights + bias
    signed_err = pred - actual
    dw = (2 / n_days) * (sales.T @ signed_err)
    db = 2 * np.mean(signed_err)

    weights = weights - (dw * learning_rate)
    bias -= (db * learning_rate)
    pred = sales @ weights + bias
    signed_err = pred - actual

    return weights, float(bias), float(np.mean(signed_err ** 2))


def fit_parameters(sales: np.ndarray, weights: np.ndarray, bias: float,
                   actual: np.ndarray, learning_rate: float, steps: int
                   ) -> Tuple[np.ndarray, float]:
    """3 — Carry a vector of parameters through a loop (~8 minutes).

    Learn:
    Repeat the joint update exactly steps times. In each iteration, all
    days contribute to the gradients. There is no loop over individual
    products or days: NumPy handles those calculations together.
    Recompute errors and gradients using the current weights and bias.
    Do not reset parameters or reuse initial gradients inside the loop.

    Protect the caller's weights. Assignment alone (current = weights)
    makes another name for the same array. If you later use -= on that
    array, the caller's data changes too. Start with an independent copy,
    or consistently create new arrays through non-in-place arithmetic.
    Return an independent weight array even when steps=0; in that case its
    values and bias stay at their starting values. No early stopping today.

    Independent API example — copying an array:
        original = np.array([4., 7.])
        working = original.copy()
        working[0] = 9.
        print(original)  # [4., 7.]
        print(working)   # [9., 7.]

    Task: Return an independent weight array and bias after exactly steps updates.
    Example input/output:
        sales = np.array([[0, 0], [1, 0], [0, 2]])
        weights = np.array([1., 1.])
        bias = 0.
        actual = np.array([1., 3., 7.])
        learning_rate = 0.15
        steps = 2
        Expected output: (np.array([1.3, 2.44]), 1.14)

    Pause: with zero updates, do your returned weights still share input memory?
    """
    weights = weights.copy()
    pred = sales @ weights + bias
    signed_err = pred - actual
    n_days = sales.shape[0]

    for _ in range(steps):
        dw = (2 / n_days) * (sales.T @ signed_err)
        db = 2 * np.mean(signed_err)
        weights = weights - (dw * learning_rate)
        bias -= (db * learning_rate)
        pred = sales @ weights + bias
        signed_err = pred - actual

    return weights, float(bias)


def training_history(sales: np.ndarray, weights: np.ndarray, bias: float,
                     actual: np.ndarray, learning_rate: float, steps: int
                     ) -> Tuple[np.ndarray, float, np.ndarray]:
    """4 — See the same loss history with vector weights (~8 minutes).

    Learn:
    Repeat the training loop and record its progress. History entry 0 is
    MSE before any updates. Entry k is MSE after k joint updates, so the
    array contains steps+1 values. The final history value must describe
    the returned weights and bias, not the model from the previous step.

    Each recorded post-update loss needs predictions from BOTH updated
    parameters. Initial data are unchanged, but predictions are not.
    Return independent weights, including for zero updates. With zero
    steps, return a history with only the initial loss.

    Independent API reminder:
        readings = [10.]
        readings.append(6.)
        print(np.array(readings, dtype=float))  # [10., 6.]
    Low training loss describes fit to these observations; it does not
    guarantee accuracy on future days. Today focus on correct recording.

    Task: Return final weights, bias, and MSE history including the initial MSE.
    Example input/output:
        sales = np.array([[0, 0], [1, 0], [0, 2]])
        weights = np.array([1., 1.])
        bias = 0.
        actual = np.array([1., 3., 7.])
        learning_rate = 0.15
        steps = 2
        Expected output: (np.array([1.3, 2.44]), 1.14,
                          np.array([10., 1.96, 0.4312]))

    Pause: why do two training updates produce three recorded losses?
    """
    weights = weights.copy()
    pred = sales @ weights + bias
    signed_err = pred - actual
    n_days = sales.shape[0]
    mse_history = [np.mean(signed_err ** 2)]

    for _ in range(steps):
        dw = (2 / n_days) * (sales.T @ signed_err)
        db = 2 * np.mean(signed_err)
        weights = weights - (dw * learning_rate)
        bias -= (db * learning_rate)
        pred = sales @ weights + bias
        signed_err = pred - actual
        mse_history.append(np.mean(signed_err ** 2))

    return weights, float(bias), np.array(mse_history, dtype=float)


def inspect_errors(day_ids: np.ndarray, sales: np.ndarray, weights: np.ndarray,
                   bias: float, actual: np.ndarray, threshold: float
                   ) -> Tuple[float, np.ndarray, np.ndarray]:
    """5 — Review: which days still have large errors? (~7 minutes).

    Learn:
    Use the supplied parameters as they are: this function EVALUATES a
    model; it does not train or update anything. Calculate overall MSE
    using ALL days, then identify days whose absolute error is too large.
    Return their signed errors so we can distinguish overprediction from
    underprediction. A small overall MSE can still hide one poor prediction.

    The same 1D mask selects IDs and errors, preserving their alignment
    and original order. A value exactly equal to threshold is not above it.
    If no day qualifies, the selected arrays are empty, while overall MSE
    is still defined because the original dataset is nonempty.

    Independent API reminders:
        print(np.abs(np.array([-2., 1.])))  # [2., 1.]
        labels = np.array([8, 9, 10])
        print(labels[np.array([False, True, True])])  # [9, 10]

    Task: Return overall MSE, IDs with absolute error above threshold, and their signed errors.
    Example input/output:
        day_ids = np.array([101, 102, 103])
        sales = np.array([[0, 0], [1, 0], [0, 2]])
        weights = np.array([1.3, 2.44])
        bias = 1.14
        actual = np.array([1., 3., 7.])
        threshold = 0.5
        Expected output: (0.4312, np.array([102, 103]), np.array([-0.56, -0.98]))

    Pause: is overall MSE computed before or after filtering out smaller errors?
    """
    pred = sales @ weights + bias
    signed_err = pred - actual
    mse = float(np.mean(signed_err ** 2))

    keep = np.abs(signed_err) > threshold
    return mse, day_ids[keep], signed_err[keep]
