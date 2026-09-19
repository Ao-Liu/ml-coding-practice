"""Day 7 — Repeat one bias update until it becomes a training loop.

Five exercises, about 35–45 minutes. Same one-product shop as Day 6:
  prediction = units * weight + bias
Only bias changes. Keep weight and the data fixed throughout every run.
No transpose, matrix multiplication, or new gradient formula today.

Recall: error = prediction - actual; MSE = mean(error squared);
        db = 2 * mean(error); new_bias = bias - learning_rate * db.

units and actual are nonempty 1D arrays of the same length. units contains
nonnegative integers; actual contains floats. Scalar inputs are finite
floats; learning rates are positive; steps is a nonnegative integer.
Assume valid inputs and finite intermediate values for these small runs.
Do not modify input arrays. Return Python floats for scalar results and
floating-point NumPy arrays for histories or collections of results.

Write operations directly again rather than calling earlier exercises.
Loops over training steps are allowed from exercise 3 onward; keep
prediction, errors, and means as NumPy operations across ALL days.
Each iteration uses the entire dataset, not one day at a time.

From the repository root, start with:
  python -m pytest -q week01_numpy_logreg/day07_bias_training -k one_update
Replace the name after -k to test your current function. Implementations
are intentionally blank; unfinished exercises will fail their tests.
"""

from typing import Tuple
import numpy as np


def one_update(units: np.ndarray, weight: float, bias: float,
               actual: np.ndarray, learning_rate: float) -> Tuple[float, float]:
    """1 — Warm-up: repeat one bias update (~5 minutes).

    Learn:
    Predict with the current bias, then use signed errors to measure its
    slope: db = 2 * mean(error). Move against that slope by subtracting
    learning_rate * db from the OLD bias. This is the same step as Day 6.
    The gradient is not the new bias, and it is not twice the MSE.
    To measure the updated model, its predictions must be computed again.

    Small reasoning example: if db=-2 and the learning rate is 0.1,
    the amount subtracted is -0.2, so bias goes UP by 0.2.
    Independent API reminder:
        print(float(np.mean(np.array([3., 7.]))))  # 5.0

    Task: Return the bias after one update and that updated model's MSE.
    Example input/output:
        units = np.array([1, 3])
        weight = 2.
        bias = 0.
        actual = np.array([5., 9.])
        learning_rate = 0.1
        Expected output: (0.6, 5.76)

    Pause: which predictions belong to the gradient, and which to the loss?
    """
    prediction = units * weight + bias
    signed_err = prediction - actual
    db = 2 * np.mean(signed_err)
    bias = bias - db * learning_rate
    new_pred = units * weight + bias
    mse = np.mean((new_pred - actual) ** 2)
    return float(bias), float(mse)


def two_updates(units: np.ndarray, weight: float, bias: float,
                actual: np.ndarray, learning_rate: float) -> Tuple[float, float]:
    """2 — Do it twice, without a loop yet (~7 minutes).

    Learn:
    The first update changes the model, so the second update starts from
    that NEW bias. Its predictions, errors, and db must all be recomputed.
    Reusing the first db would mean following an old slope at a new place.

    Think of walking downhill: after moving, check the slope where you
    are now. If you are closer to the bottom, a smaller gradient may ask
    for a smaller move, even though the learning rate stays the same.
    Keep weight, units, actual, and learning_rate unchanged. Only the
    current bias and quantities calculated from it change.
    Write two update blocks directly today; repetition makes the boundary
    between the first and second update visible.

    Independent Python reminder — assignment changes the local value:
        value = 4.
        value = value + 1.
        print(value)  # 5.0; later expressions see this new value

    Task: Return the bias after exactly two updates and its MSE.
    Example input/output:
        units = np.array([1, 3])
        weight = 2.
        bias = 0.
        actual = np.array([5., 9.])
        learning_rate = 0.1
        Expected output: (1.08, 3.6864)

    Pause: should the second update start from 0 or from 0.6?
    """
    prediction = units * weight + bias
    signed_err = prediction - actual

    db = 2 * np.mean(signed_err)
    bias = bias - db * learning_rate
    prediction = units * weight + bias
    signed_err = prediction - actual

    # when b changes, db also changes, the model changes. Its predictions, errors, and db must all be recomputed.
    db = 2 * np.mean(signed_err)
    bias = bias - db * learning_rate
    prediction = units * weight + bias
    signed_err = prediction - actual
    mse = np.mean(signed_err ** 2)
    return float(bias), float(mse)


def fit_bias(units: np.ndarray, weight: float, bias: float,
             actual: np.ndarray, learning_rate: float, steps: int) -> float:
    """3 — Replace repeated blocks with a loop (~7 minutes).

    Learn:
    Each iteration is one complete update. The current bias carries
    forward into the next iteration; do not reset it inside the loop.
    Recalculate predictions, errors, and db each time using this current
    bias. They describe a changing model, even though the data stay fixed.
    You are repeating yesterday's operation, not introducing new math.

    Independent Python example — repeat an operation a fixed number of times:
        value = 1
        for _ in range(3):
            value = value * 2
        print(value)  # 8
    range(0) performs no iterations. With zero training steps, return
    the original bias. Do not add an early-stopping rule today.

    Task: Return the bias after exactly steps gradient updates.
    Example input/output:
        units = np.array([1, 3])
        weight = 2.
        bias = 0.
        actual = np.array([5., 9.])
        learning_rate = 0.1
        steps = 3
        Expected output: 1.464

    Pause: what is fixed during training, and what changes each iteration?
    """
    prediction = units * weight + bias
    signed_err = prediction - actual
    db = 2 * np.mean(signed_err)

    for _ in range(steps):
        bias -= (db * learning_rate)
        prediction = units * weight + bias
        signed_err = prediction - actual
        db = 2 * np.mean(signed_err)

    return float(bias)



def loss_history(units: np.ndarray, weight: float, bias: float,
                 actual: np.ndarray, learning_rate: float, steps: int
                 ) -> Tuple[float, np.ndarray]:
    """4 — Watch learning happen (~8 minutes).

    Learn:
    The final loss alone hides the path. A history records snapshots:
    entry 0 is MSE BEFORE training; entry 1 is MSE AFTER the first update;
    entry 2 is MSE AFTER the second update, and so on.
    Therefore, steps updates produce steps+1 entries, not steps entries.
    A post-update loss uses post-update predictions. Old predictions do
    not automatically change when you assign a new value to bias.

    Independent API example — build a list, then convert it to an array:
        readings = [4.]
        readings.append(2.5)
        readings.append(1.)
        print(np.array(readings, dtype=float))  # [4., 2.5, 1.]
    This is just storage: the numbers you store here will be measured MSEs.
    For zero updates, the history still contains the initial loss.

    Task: Return the final bias and MSE history, including the initial MSE.
    Example input/output:
        units = np.array([1, 3])
        weight = 2.
        bias = 0.
        actual = np.array([5., 9.])
        learning_rate = 0.1
        steps = 3
        Expected output: (1.464, np.array([9., 5.76, 3.6864, 2.359296]))

    Pause: if db is zero but MSE is positive, could a shared bias alone
    remove the remaining errors? Opposite signed errors can cancel.
    """
    prediction = units * weight + bias
    signed_err = prediction - actual
    db = 2 * np.mean(signed_err)
    mse_history = [float(np.mean(signed_err ** 2))]

    for _ in range(steps):
        bias -= (db * learning_rate)
        prediction = units * weight + bias
        signed_err = prediction - actual
        db = 2 * np.mean(signed_err)
        mse_history.append(float(np.mean(signed_err ** 2)))

    # return float(bias), np.array(mse_history, dtype=float)
    return float(bias), np.array(mse_history, dtype=float)


def compare_learning_rates(units: np.ndarray, weight: float, bias: float,
                           actual: np.ndarray, learning_rates: np.ndarray,
                           steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """5 — Same starting point, different step sizes (~8 minutes).

    Learn:
    A learning rate scales each gradient step. A small rate may approach
    a good bias slowly. A large one may jump past it and even increase loss.
    Compare fairly: each rate gets the SAME initial bias, data, weight,
    and number of updates. Do not start run 2 from run 1's final bias.
    Keep a separate current bias for each run; the original bias is the
    common starting point. A loop over rates and a loop over steps are
    appropriate here. No new vectorization trick is needed.

    Independent Python example — resetting inside an outer loop:
        for label in ['first', 'second']:
            counter = 0
            for _ in range(2):
                counter += 1
            print(label, counter)  # first 2; second 2 (not 4)
    Collect results in the order rates were supplied; do not sort them.
    learning_rates is a nonempty 1D float array. Each returned array has
    that same length. With zero steps, every run keeps the initial model.

    Task: Return final biases and final MSEs for the supplied learning rates.
    Example input/output:
        units = np.array([1, 3])
        weight = 2.
        bias = 0.
        actual = np.array([5., 9.])
        learning_rates = np.array([0.1, 0.5, 1.1])
        steps = 2
        Expected output: (np.array([1.08, 3., -1.32]),
                          np.array([3.6864, 0., 18.6624]))

    Pause: the largest rate has a worse final loss than the initial 9.
    After passing, run your loss_history for each rate to see its path.
    These rates apply to this fixed-weight, bias-only MSE example; they
    are not universal good/bad rates for models that also train weights.
    """
    init_prediction = units * weight + bias
    init_signed_err = init_prediction - actual
    init_db = 2 * np.mean(init_signed_err)
    bias_history = []
    mse_history = []

    for rate in learning_rates:
        db = init_db
        b = bias
        signed_err = init_signed_err
        for _ in range(steps):
            b -= (db * rate)
            prediction = units * weight + b
            signed_err = prediction - actual
            db = 2 * np.mean(signed_err)
        bias_history.append(float(b))
        mse_history.append(float(np.mean(signed_err ** 2)))

    return np.array(bias_history, dtype=float), np.array(mse_history, dtype=float)

