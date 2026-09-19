"""Day 6 — One product, one small change at a time (35–45 minutes).

A shop sells ONE product. units contains the units sold on each day:
  prediction = units * weight + bias
weight is the model's contribution per unit; bias is a daily baseline.
Both are scalar floats today. units and actual are 1D arrays, shape (N,).
There is no matrix multiplication, transpose, or training loop today.

Work from top to bottom. Each exercise repeats familiar arithmetic and
adds one idea. Before running a test, predict what should happen in words.
Write the operations again instead of calling earlier exercise functions.

All arrays are nonempty, finite, and have matching lengths. units contains
nonnegative integers; actual contains floats. Other inputs are finite
floats; step and learning_rate are positive. Do not modify inputs.
Return floating-point arrays for predictions and Python floats for scalars.

Run one exercise from the repository root:
  python -m pytest -q week01_numpy_logreg/day06_bias_intuition -k change_bias
Replace change_bias with the function you are practicing. Unfinished
functions intentionally raise NotImplementedError.
"""

from typing import Tuple
import numpy as np


def change_bias(units: np.ndarray, weight: float, bias: float,
                increase: float) -> Tuple[np.ndarray, np.ndarray]:
    """1 — What changes when bias increases? (~5 minutes)

    Learn:
    Bias is added AFTER units are multiplied by weight. It is shared by
    every day, regardless of how many units were sold. Keep weight fixed.
    If a prediction was 12 and bias increases by 0.5, it becomes 12.5.
    A day with zero sales also gets this extra 0.5: bias does not depend
    on sales. There is no loss or gradient to calculate in this exercise.

    Independent API example — a scalar acts on every array element:
        values = np.array([3., 8., 0.])
        print(values + 2.)  # [5., 10., 2.]
    A tuple can hold two arrays; its order matters to the caller.

    Task: Return the predictions before and after increasing only bias.
    Example input/output:
        units = np.array([1, 3])
        weight = 2.
        bias = 0.
        increase = 1.
        Expected output: (np.array([2., 6.]), np.array([3., 7.]))

    Pause: how much did each day's prediction change, and why equally?
    """
    before = units * weight + bias
    bias += increase
    after = units * weight + bias
    return before, after


def change_weight(units: np.ndarray, weight: float, bias: float,
                  increase: float) -> Tuple[np.ndarray, np.ndarray]:
    """2 — What changes when weight increases? (~5 minutes)

    Learn:
    Weight is multiplied by units, so its effect depends on the day.
    If weight increases by 0.5, a day with 2 units gains 1 in predicted
    revenue; a day with 6 units gains 3. A day with zero units gains 0.
    Bias stays fixed. Contrast this with exercise 1, where every day
    changed equally. The distinction will later explain weight gradients;
    today you only need to see the change in predictions.

    Independent API example — a scalar scales each value:
        values = np.array([2., 5., 0.])
        print(values * 3.)  # [6., 15., 0.]
    weight is a number, not an array: ordinary * is appropriate here.

    Task: Return the predictions before and after increasing only weight.
    Example input/output:
        units = np.array([1, 3])
        weight = 2.
        bias = 0.
        increase = 1.
        Expected output: (np.array([2., 6.]), np.array([3., 9.]))

    Pause: why did the second prediction change more than the first?
    """
    predicted_before = units * weight + bias
    weight += increase
    predicted_after = units * weight + bias
    return predicted_before, predicted_after


def compare_biases(units: np.ndarray, weight: float, bias: float,
                   actual: np.ndarray, step: float) -> Tuple[float, float, float]:
    """3 — Try three nearby biases before using gradients (~10 minutes).

    Learn:
    Raising a prediction is only useful if it gets closer to actual.
    Signed error = prediction - actual; MSE = mean(error squared).
    For example, with actual=8, a prediction of 5 has squared error 9;
    a prediction of 6 has squared error 4. Moving upward helped there.
    But if prediction was already 10, moving to 11 would make it worse.

    Keep weight fixed and evaluate three separate choices:
    bias-step, bias, and bias+step. Each needs its OWN predictions and MSE.
    step is the distance between candidates, not a learning rate.
    Repeat the familiar calculations three times; no loop or clever
    broadcasting is needed. Return losses in candidate order, not sorted.

    Independent API examples:
        print(np.array([-2., 3.]) ** 2)  # [4., 9.]
        print(float(np.mean(np.array([2., 6.]))))  # 4.0

    Task: Return the MSEs for bias-step, bias, and bias+step, in that order.
    Example input/output:
        units = np.array([1, 3])
        weight = 2.
        bias = 0.
        actual = np.array([5., 9.])
        step = 1.
        Expected output: (16.0, 9.0, 4.0)

    Pause: which direction helped here? You found it without a derivative.
    """
    # bias - step
    prediction_bs1 = units * weight + (bias - step)
    signed_err_bs1 = prediction_bs1 - actual
    mse_bs1 = np.mean(signed_err_bs1 ** 2)
    # bias
    prediction_b = units * weight + bias
    signed_err_b = prediction_b - actual
    mse_b = np.mean(signed_err_b ** 2)
    # bias + step
    prediction_bs2 = units * weight + (bias + step)
    signed_err_bs2 = prediction_bs2 - actual
    mse_bs2 = np.mean(signed_err_bs2 ** 2)
    return float(mse_bs1), float(mse_b), float(mse_bs2)


def bias_slope(units: np.ndarray, weight: float, bias: float,
               actual: np.ndarray) -> float:
    """4 — A gradient describes the local direction of loss (~10 minutes).

    Learn:
    Exercise 3 checked loss at three places. A derivative describes how
    loss changes for a TINY increase in bias at the current place.
    Call this number db. It is NOT the loss, NOT the new bias, and NOT
    the amount you should move. It measures loss's local rate of change.

    For one day, let e = prediction - actual. Increasing bias by a tiny
    amount increases e by the same amount. The derivative of e squared
    with respect to e is 2*e; e changes at rate 1 with respect to bias.
    Thus that day's loss derivative is 2*e. MSE averages across days,
    so db = 2 * mean(signed errors). The square has already been
    differentiated: do not square errors again or take absolute values.

    If a day's prediction is 3 below actual, its signed error is -3:
    its loss derivative is -6, meaning a small increase in bias helps.
    Averaging signed errors combines all days, including disagreements.
    Positive db: increasing bias locally raises loss. Negative db:
    increasing bias locally lowers loss. Zero db: no first-order change.

    Independent API example — signs can cancel in a mean:
        print(np.mean(np.array([-4., 4.])))  # 0.0

    Task: Return the MSE derivative with respect to bias at the current model.
    Example input/output:
        units = np.array([1, 3])
        weight = 2.
        bias = 0.
        actual = np.array([5., 9.])
        Expected output: -6.0

    Pause: does this sign agree with the direction that helped in exercise 3?
    """
    p = units * weight + bias
    e = p - actual
    db = 2 * np.mean(e)
    return float(db)


def update_bias_once(units: np.ndarray, weight: float, bias: float,
                     actual: np.ndarray, learning_rate: float
                     ) -> Tuple[float, float, float]:
    """5 — Take one small step and check its effect (~10 minutes).

    Learn:
    Keep weight fixed. First distinguish four quantities:
      bias: where the parameter is now;
      db: the loss slope there, equal to 2*mean(signed errors);
      learning_rate*db: the amount to SUBTRACT;
      new_bias: where the parameter lands after subtracting that amount.

    We want lower loss, so move against its slope:
      new_bias = bias - learning_rate * db
    If db is positive, this lowers bias. If db is negative, subtracting
    a negative number raises bias. Subtraction does NOT always mean
    the new parameter is smaller. For instance, starting at 2 with
    db=-4 and learning_rate=0.1 gives 2-(-0.4)=2.4.

    Measure MSE once with the old bias and again with the new bias.
    The second measurement needs fresh predictions. A large learning
    rate can overshoot: always measure rather than assume improvement.
    Today this is ONE bias update, not a weight update or a training loop.

    Independent API example — save a result without altering its input:
        values = np.array([5., 7.])
        shifted = values - 2.
        print(values)   # [5., 7.]
        print(shifted)  # [3., 5.]

    Task: Return the updated bias, old MSE, and new MSE after one bias step.
    Example input/output:
        units = np.array([1, 3])
        weight = 2.
        bias = 0.
        actual = np.array([5., 9.])
        learning_rate = 0.1
        Expected output: (0.6, 9.0, 5.76)

    Pause: explain why bias increased, then check whether loss decreased.
    """
    prediction_before = units * weight + bias
    signed_err_before = prediction_before - actual
    mse_before = np.mean(signed_err_before ** 2)
    db = 2 * np.mean(signed_err_before)
    bias -= db * learning_rate
    prediction_after = units * weight + bias
    signed_err_after = prediction_after - actual
    mse_after = np.mean(signed_err_after ** 2)
    return float(bias), float(mse_before), float(mse_after)
