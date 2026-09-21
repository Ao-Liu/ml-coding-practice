"""Day 8 — One product: learn weight AND bias (35–45 minutes).

Same shop, same model: prediction = units * weight + bias.
weight and bias are scalar floats, not arrays. units and actual are
nonempty 1D arrays of matching length. No matrix multiplication today.

1: compare weights; 2: understand dw; 3: update only weight;
4: update both parameters once; 5: repeat and record loss.
Write the NumPy operations again in each function, without calling earlier
exercise functions. Use a loop over updates only in exercise 5.

Assume finite valid inputs: nonnegative integer units, float actual values,
finite scalar parameters, positive step/learning_rate, and steps >= 0.
Intermediate values in tests stay finite. Do not modify input arrays.
Scalar results should be Python floats; loss history is a float ndarray.

Always use signed error = prediction - actual.
MSE = mean(error squared), with NO extra factor of 1/2.
The mathematical formulas are provided; Learn explains their meaning.
The API snippets teach separate operations, not full task solutions.

From the repository root:
  python -m pytest -q week02_linear_models/day08_weight_and_bias -k compare_weights
Unfinished functions intentionally raise NotImplementedError.
"""

from typing import Tuple
import numpy as np


def compare_weights(units: np.ndarray, weight: float, bias: float,
                    actual: np.ndarray, step: float) -> Tuple[float, float, float]:
    """1 — Observe a direction before calculating a gradient (~6 minutes).

    Learn:
    Yesterday only bias changed. Today first hold bias fixed and try
    three weights: weight-step, weight, and weight+step. Each describes a
    separate model and needs its own predictions and MSE.

    Weight multiplies units. Increasing weight by 0.5 raises a prediction
    by 1 on a day with 2 units, but by 3 on a day with 6 units. Bias would
    raise them equally. A day with zero sales is unaffected by weight.
    Whether the increase helps depends on the actual revenue: compare
    losses, rather than assuming that larger predictions are better.

    Independent API reminders:
        print(np.array([2., 4.]) * 0.5)  # [1., 2.]
        print(np.array([-2., 3.]) ** 2)  # [4., 9.]
    Repeat the three calculations directly; no loop is needed.

    Task: Return MSEs for weight-step, weight, and weight+step, in that order.
    Example input/output:
        units = np.array([0, 2])
        weight = 1.
        bias = 0.
        actual = np.array([1., 5.])
        step = 1.
        Expected output: (13.0, 5.0, 1.0)

    Pause: why is the zero-sales day's prediction unchanged in all three?
    """
    prediction_1 = units * (weight - step) + bias
    signed_err_1 = prediction_1 - actual
    mse_1 = float(np.mean(signed_err_1 ** 2))

    prediction_2 = units * weight + bias
    signed_err_2 = prediction_2 - actual
    mse_2 = float(np.mean(signed_err_2 ** 2))

    prediction_3 = units * (weight + step) + bias
    signed_err_3 = prediction_3 - actual
    mse_3 = float(np.mean(signed_err_3 ** 2))

    return mse_1, mse_2, mse_3



def weight_slope(units: np.ndarray, weight: float, bias: float,
                 actual: np.ndarray) -> float:
    """2 — Why the weight gradient includes units (~8 minutes).

    Learn:
    A gradient measures how MSE changes for a tiny increase in a parameter.
    For bias, every prediction changes at rate 1, so db = 2*mean(error).
    For weight, a day's prediction changes at rate units for that day.

    Consider two days with the SAME signed error of +2. If their sales
    are 1 and 4 units, increasing weight by 0.01 raises their predictions
    by 0.01 and 0.04. Both were too high already; the second day's loss is
    more sensitive to this parameter because more units multiply it.

    For one day: d(error squared)/dw = 2 * error * units.
    The 2 comes from the square; units comes from how prediction changes
    with weight. Averaging over days gives dw = 2 * mean(units * error).
    Multiply EACH day's error by THAT day's units before averaging.
    Do not use mean(units)*mean(error): that mixes different days together.
    Return the slope only; do not update any parameter yet.

    Independent API example — multiplication pairs matching positions:
        a = np.array([2., 5.])
        b = np.array([-1., 3.])
        print(a * b)  # [-2., 15.], still one entry per position

    Task: Return the current MSE gradient with respect to weight.
    Example input/output:
        units = np.array([0, 2])
        weight = 1.
        bias = 0.
        actual = np.array([1., 5.])
        Expected output: -6.0

    Pause: does the sign agree with the helpful direction in exercise 1?
    """
    prediction = units * weight + bias
    signed_err = prediction - actual # like [1, 2, 3]
    dw = 2 * np.mean(signed_err * units)
    return float(dw)


def update_weight_once(units: np.ndarray, weight: float, bias: float,
                       actual: np.ndarray, learning_rate: float
                       ) -> Tuple[float, float, float]:
    """3 — Same update rule, now for weight (~6 minutes).

    Learn:
    Keep bias fixed. Calculate dw = 2*mean(units*error) using the old
    model. Then new_weight = weight - learning_rate*dw.
    This is exactly yesterday's update rule applied to another parameter.
    The gradient is NOT the new weight: start from the old weight.

    If weight is 3, dw is +2, and learning_rate is 0.1, the new weight
    is 2.8. A positive slope asks for a decrease to try to lower MSE.
    A negative slope asks for an increase. Large steps can still overshoot.
    Evaluate loss before the change, then make fresh predictions with the
    new weight to measure loss after it. Bias remains unchanged.

    Independent API reminder:
        print(float(np.mean(np.array([4., 8.]))))  # 6.0

    Task: Return the new weight, MSE before, and MSE after one weight update.
    Example input/output:
        units = np.array([0, 2])
        weight = 1.
        bias = 0.
        actual = np.array([1., 5.])
        learning_rate = 0.1
        Expected output: (1.6, 5.0, 2.12)

    Pause: can changing weight fix the prediction for the zero-sales day?
    """
    prediction = units * weight + bias
    signed_err = prediction - actual
    dw = 2 * np.mean(signed_err * units)
    mse_before = float(np.mean(signed_err ** 2))

    weight -= (dw * learning_rate)
    prediction = units * weight + bias
    signed_err = prediction - actual
    mse_after = float(np.mean(signed_err ** 2))
    return float(weight), mse_before, mse_after


def update_both_once(units: np.ndarray, weight: float, bias: float,
                     actual: np.ndarray, learning_rate: float
                     ) -> Tuple[float, float, float, float]:
    """4 — Two gradients from ONE model state (~8 minutes).

    Learn:
    Now let both parameters move. Bias can shift every prediction; weight
    changes predictions in proportion to sales. Their roles complement
    each other, so the model can adjust both its baseline and its slope.

    Use the familiar formulas:
      dw = 2 * mean(units * error)
      db = 2 * mean(error)
    Both use the SAME signed-error array from the old weight and old bias.
    Calculate both gradients before changing either parameter. Updating
    weight and then recomputing errors for db would describe two different
    model states, not the simultaneous gradient step requested here.
    Update each parameter by subtracting learning_rate times its gradient.
    Only after both updates, predict again to measure the new MSE.

    Independent Python reminder — a calculated value does not auto-update:
        x = 3.
        doubled = 2 * x
        x = 4.
        print(doubled)  # 6.0, not 8.0

    Task: Return new weight, new bias, old MSE, and new MSE after one joint update.
    Example input/output:
        units = np.array([0, 2])
        weight = 1.
        bias = 0.
        actual = np.array([1., 5.])
        learning_rate = 0.1
        Expected output: (1.6, 0.4, 5.0, 1.16)

    Pause: compare the new MSE with exercise 3; which extra parameter moved?
    """
    prediction = units * weight + bias
    signed_err = prediction - actual
    mse_before = np.mean(signed_err ** 2)

    dw = 2 * np.mean(signed_err * units)
    weight -= (dw * learning_rate)

    db = 2 * np.mean(signed_err)
    bias -= (db * learning_rate)

    prediction = units * weight + bias
    signed_err = prediction - actual
    mse_after = np.mean(signed_err ** 2)

    return float(weight), float(bias), float(mse_before), float(mse_after)





def fit_line(units: np.ndarray, weight: float, bias: float,
             actual: np.ndarray, learning_rate: float, steps: int
             ) -> Tuple[float, float, np.ndarray]:
    """5 — Yesterday's loop, with two changing parameters (~10 minutes).

    Learn:
    Repeat the joint update from exercise 4. At the start of each step,
    the current weight and bias form one model. Recompute its predictions
    and errors, then both gradients. Carry both updated parameters into
    the next iteration; never reset them inside the loop.

    Record the initial MSE and a fresh MSE after every update. As yesterday,
    steps updates mean steps+1 history entries. Zero steps returns the
    initial parameters and a history containing just the initial loss.
    units, actual, and learning_rate stay fixed throughout the run.

    Independent API reminder — record values and convert to an array:
        readings = [8.]
        readings.append(5.)
        print(np.array(readings, dtype=float))  # [8., 5.]
    Do not stop early or choose a learning rate automatically today.
    Rates that worked for bias alone need not work when weight moves too.

    Task: Return final weight, final bias, and MSE history after exactly steps joint updates.
    Example input/output:
        units = np.array([0, 2])
        weight = 1.
        bias = 0.
        actual = np.array([1., 5.])
        learning_rate = 0.1
        steps = 2
        Expected output: (1.88, 0.6, np.array([5., 1.16, 0.2848]))

    After passing, try more steps on this same dataset: the line that fits
    both observations has weight=2 and bias=1. Observe how close you get;
    do not hard-code those values or expect every dataset to fit perfectly.
    """
    init_pred = units * weight + bias
    init_signed_err = init_pred - actual
    init_dw = 2 * np.mean(init_signed_err * units)
    init_db = 2 * np.mean(init_signed_err)
    init_mse = np.mean(init_signed_err ** 2)
    mse_history = [init_mse]

    mse = init_mse
    dw = init_dw
    db = init_db
    for _ in range(steps):
        weight -= (dw * learning_rate)
        bias -= (db * learning_rate)
        pred = weight * units + bias
        signed_err = pred - actual
        dw = 2 * np.mean(signed_err * units)
        db = 2 * np.mean(signed_err)
        mse = np.mean(signed_err ** 2)
        mse_history.append(mse)

    return float(weight), float(bias), np.array(mse_history, dtype=float)

