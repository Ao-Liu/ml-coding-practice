"""Day 9 — Two products, two separate weights (35–45 minutes).

One row of sales is one day. Column 0 is product A; column 1 is product B.
sales has shape (N,2). Each row still has ONE observed total revenue in
actual, shape (N,). weight_a, weight_b, and bias are scalar floats.

The model adds three contributions:
  prediction = A's units * weight_a + B's units * weight_b + bias
There is one shared daily baseline, so add bias only once.

Today write the two columns' calculations separately. Do not use @ or .T;
we will connect this version to matrix operations after it makes sense.
Repeat the NumPy operations directly rather than calling earlier exercises.
Only exercise 5 needs a loop, over training steps rather than data rows.

Assume valid, nonempty finite inputs: nonnegative integer sales, float actual,
finite parameters, positive learning_rate, and integer steps >= 0. Examples
use three days so the two columns cannot be confused with the number of days.
Do not modify input arrays. Return floating-point NumPy arrays for predictions
and histories, and Python floats for scalar results. MSE = mean(error^2).

From the repository root:
  python -m pytest -q week02_linear_models/day09_two_products -k predict_two_products
All explanations are in this file. Implement one function at a time;
unfinished functions intentionally raise NotImplementedError.
"""

from typing import Tuple
import numpy as np


def predict_two_products(sales: np.ndarray, weight_a: float,
                         weight_b: float, bias: float) -> np.ndarray:
    """1 — One prediction from two contributions (~5 minutes).

    Learn:
    Yesterday you multiplied one product's units by one weight. Now do
    that separately for A and B, then add their contributions and one bias.
    The two products do NOT produce separate final predictions: their
    contributions belong to the same day's total revenue.

    For example, if a day sells 2 A and 3 B, weights are 4 and 5, and
    bias is 1, its prediction is 2*4 + 3*5 + 1 = 24. This is just the
    same rule with one more term; no new matrix operation is needed.

    Independent API reminder — selecting a column:
        grid = np.array([[8, 9], [4, 5], [2, 7]])
        print(grid[:, 1])  # [9, 5, 7], shape (3,)
    ':' keeps every row. The second index chooses the column (zero-based).
    Each extracted column has shape (N,), so its entries line up by day.

    Task: Return one predicted total revenue per day using the two sales columns.
    Example input/output:
        sales = np.array([[0, 0], [1, 0], [0, 2]])
        weight_a = 1.
        weight_b = 1.
        bias = 0.
        Expected output: np.array([0., 1., 2.])

    Pause: why does the result have three values, rather than two?
    """
    sales_a = sales[:,0]
    sales_b = sales[:,1] # # of b sold on day 1, 2, 3
    pa = sales_a * weight_a
    pb = sales_b * weight_b
    return pa + pb + bias


def change_a_weight(sales: np.ndarray, weight_a: float, weight_b: float,
                    bias: float, increase: float) -> Tuple[np.ndarray, np.ndarray]:
    """2 — Change A's weight, hold everything else fixed (~5 minutes).

    Learn:
    Increasing weight_a changes predictions according to A's sales ONLY.
    B's weight and the bias stay fixed. If A sold 0 units that day, this
    change cannot affect that day's prediction, even if B sold many units.
    With an increase of 0.5, a day selling 4 A gains 2 in prediction;
    a day selling 1 A gains 0.5. This repeats Day 6 with another fixed term.

    Independent API reminder:
        values = np.array([2., 0., 5.])
        print(values * 0.5)  # [1., 0., 2.5]
    Calculate predictions for the original and changed models; each
    includes BOTH products and the shared bias.

    Task: Return predictions before and after increasing only weight_a.
    Example input/output:
        sales = np.array([[0, 0], [1, 0], [0, 2]])
        weight_a = 1.
        weight_b = 1.
        bias = 0.
        increase = 0.5
        Expected output: (np.array([0., 1., 2.]), np.array([0., 1.5, 2.]))

    Pause: which days changed, and what do their A sales have in common?
    """
    sales_a = sales[:,0]
    sales_b = sales[:,1]
    pa_before = sales_a * weight_a
    pb = sales_b * weight_b

    p1 = pa_before + pb + bias

    pa_after = sales_a * (weight_a + increase)
    p2 = pa_after + pb + bias
    return p1, p2


def separate_gradients(sales: np.ndarray, weight_a: float, weight_b: float,
                       bias: float, actual: np.ndarray) -> Tuple[float, float, float]:
    """3 — Same error, different influence for each weight (~10 minutes).

    Learn:
    Calculate ONE total prediction per day, then signed error = prediction
    minus actual. There are no separate 'A actual' and 'B actual' values.
    Each gradient asks how changing one parameter affects this total error.

    Suppose a day's signed error is +2, and it sold 1 A and 4 B. Increasing
    weight_a by 0.01 raises prediction by 0.01; increasing weight_b by 0.01
    raises it by 0.04. Both worsen this day's overprediction, but B's
    parameter has more influence. Each needs its own sales multiplier.

    Apply yesterday's single-weight rule separately:
      dw_a = 2 * mean(A_units * error)
      dw_b = 2 * mean(B_units * error)
      db   = 2 * mean(error)
    A_units and B_units are the two sales columns. Multiply corresponding
    days BEFORE averaging; average across N days, not across two products.
    Use the SAME signed-error array in all three formulas. Do not calculate
    A-only predictions to get dw_a: error belongs to the complete model.

    Independent API example:
        x = np.array([1., 4., 2.])
        y = np.array([-2., 3., 1.])
        print(x * y)  # [-2., 12., 2.]

    Task: Return the three current gradients in the order (dw_a, dw_b, db).
    Example input/output:
        sales = np.array([[0, 0], [1, 0], [0, 2]])
        weight_a = 1.
        weight_b = 1.
        bias = 0.
        actual = np.array([1., 3., 7.])
        Expected output: (-1.3333333333333333, -6.666666666666667,
                          -5.333333333333333)

    Pause: which parameter can change the zero-sales day's prediction?
    """
    sales_a = sales[:,0]
    sales_b = sales[:,1]
    pred = sales_a * weight_a + sales_b * weight_b + bias
    signed_err = pred - actual
    dw_a = 2 * np.mean(signed_err * sales_a)
    dw_b = 2 * np.mean(signed_err * sales_b)
    db = 2 * np.mean(signed_err)

    return float(dw_a), float(dw_b), float(db)


def update_three_parameters(sales: np.ndarray, weight_a: float,
                            weight_b: float, bias: float, actual: np.ndarray,
                            learning_rate: float
                            ) -> Tuple[float, float, float, float, float]:
    """4 — One joint update, now with three parameters (~10 minutes).

    Learn:
    Use dw_a = 2*mean(A_units*error), dw_b = 2*mean(B_units*error), and
    db = 2*mean(error). All three gradients describe the OLD model.
    Calculate them before changing any parameter, so they use the same
    predictions and errors. This is Day 8's joint update with one extra weight.

    For each parameter, new = old - learning_rate * its OWN gradient.
    A positive gradient asks for a decrease, a negative one for an increase.
    For example, parameter=2, gradient=3, rate=0.1 gives new=1.7.
    After all three updates, calculate fresh total predictions and MSE.
    Do not assume loss decreases for every possible learning rate.

    Independent Python reminder:
        a, b, c = 1., 2., 3.
        snapshot = (a, b, c)
        a = 9.
        print(snapshot)  # (1.0, 2.0, 3.0): the saved values stay unchanged

    Task: Return new weight_a, new weight_b, new bias, old MSE, and new MSE.
    Example input/output:
        sales = np.array([[0, 0], [1, 0], [0, 2]])
        weight_a = 1.
        weight_b = 1.
        bias = 0.
        actual = np.array([1., 3., 7.])
        learning_rate = 0.15
        Expected output: (1.2, 2.0, 0.8, 10.0, 1.96)

    Pause: why shouldn't you recompute errors between updating A and B?
    """
    sales_a = sales[:, 0]
    sales_b = sales[:, 1]
    pred = sales_a * weight_a + sales_b * weight_b + bias
    signed_err = pred - actual
    dw_a = 2 * np.mean(signed_err * sales_a)
    dw_b = 2 * np.mean(signed_err * sales_b)
    db = 2 * np.mean(signed_err)
    mse_before = np.mean(signed_err ** 2)

    weight_a -= dw_a * learning_rate
    weight_b -= dw_b * learning_rate
    bias -= db * learning_rate
    pred = sales_a * weight_a + sales_b * weight_b + bias
    signed_err = pred - actual
    mse_after = np.mean(signed_err ** 2)
    return float(weight_a), float(weight_b), float(bias), float(mse_before), float(mse_after)



def fit_two_products(sales: np.ndarray, weight_a: float, weight_b: float,
                     bias: float, actual: np.ndarray, learning_rate: float,
                     steps: int) -> Tuple[float, float, float, np.ndarray]:
    """5 — Repeat the update and keep the familiar history (~10 minutes).

    Learn:
    Every iteration starts with the CURRENT three parameters. Recompute
    total predictions, signed errors, and all three gradients; then update
    all three parameters. Carry their new values into the next iteration.
    The data and learning rate stay fixed. There is no early stopping today.

    Save the initial MSE, then one freshly evaluated MSE after each update.
    steps updates produce steps+1 entries. With zero updates, return the
    starting parameters and a history containing only their initial MSE.
    This is the same training loop you already wrote, now with one more
    parameter. Keep the two column calculations explicit; no @ or .T needed.

    Independent API reminder:
        readings = [6.]
        readings.append(4.)
        print(np.array(readings, dtype=float))  # [6., 4.]
    Do not reset parameters inside the loop or reuse yesterday's gradients.

    Task: Return final weight_a, weight_b, bias, and MSE history after steps updates.
    Example input/output:
        sales = np.array([[0, 0], [1, 0], [0, 2]])
        weight_a = 1.
        weight_b = 1.
        bias = 0.
        actual = np.array([1., 3., 7.])
        learning_rate = 0.15
        steps = 2
        Expected output: (1.3, 2.44, 1.14, np.array([10., 1.96, 0.4312]))

    After passing, try more steps on this dataset. Its exact fit is
    weight_a=2, weight_b=3, bias=1; observe progress instead of hard-coding
    these values. Noisy datasets need not have an exact fit.
    """
    sales_a = sales[:, 0]
    sales_b = sales[:, 1]
    pred = sales_a * weight_a + sales_b * weight_b + bias
    signed_err = pred - actual
    dw_a = 2 * np.mean(signed_err * sales_a)
    dw_b = 2 * np.mean(signed_err * sales_b)
    db = 2 * np.mean(signed_err)
    mse_history = [np.mean(signed_err ** 2)]

    for _ in range(steps):
        weight_a -= (dw_a * learning_rate)
        weight_b -= (dw_b * learning_rate)
        bias -= (db * learning_rate)
        pred = sales_a * weight_a + sales_b * weight_b + bias
        signed_err = pred - actual
        dw_a = 2 * np.mean(signed_err * sales_a)
        dw_b = 2 * np.mean(signed_err * sales_b)
        db = 2 * np.mean(signed_err)
        mse_history.append(np.mean(signed_err ** 2))

    return float(weight_a), float(weight_b), float(bias), np.array(mse_history, dtype=float)