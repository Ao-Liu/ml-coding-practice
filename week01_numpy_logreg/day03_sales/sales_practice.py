"""Day 3 — Shop sales: broadcasting → weighted sums → X @ w.

7 core exercises (~40–50 minutes), then 1 optional review.
Write NumPy operations directly in EACH function: do not call your earlier
exercise functions. Repetition is intentional. No loops are needed.

Shared example (examples show array values without the np.array wrapper):
  day_ids = np.array([101, 102, 103])
  sales = np.array([[2, 1, 0], [0, 3, 2], [1, 0, 4]])
  prices = np.array([10., 5., 2.])
Rows = days (N); columns = products (D).

All inputs have valid shapes and finite values; N,D >= 1. Sales are
nonnegative integers; prices are nonnegative floats. Do not modify inputs.
Return NumPy arrays, not lists; use floating-point arrays for money/ratios.
Type hints say ndarray; the Inputs/Output lines specify dtype and shape.

Run ONE exercise from the repository root:
  python -m pytest -q week01_numpy_logreg/day03_sales -k daily_revenue
Unfinished functions intentionally raise NotImplementedError.
"""

from typing import Tuple
import numpy as np


def daily_revenue(sales: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """1 — Repeat broadcasting and axis.

    Learn: * multiplies matching elements; sum reduces an array.
    API examples (independent operations):
        a = np.array([2., 3., 4.])
        b = np.array([5., 2., 1.])
        print(a * b)  # [10., 6., 4.]

        grid = np.array([[1, 2], [3, 4], [5, 6]])
        print(np.sum(grid, axis=0))  # [9, 12], shape (2,)
        # axis chooses the dimension to reduce; choose it for your task.

    Inputs: sales: int ndarray (N,D); prices: float ndarray (D,).
    Output: float ndarray (N,).
    Task: Return total revenue for each day using multiplication and sum.
    Example input/output:
        sales = np.array([[2, 1, 0], [0, 3, 2], [1, 0, 4]])
        prices = np.array([10., 5., 2.])
        Expected output: np.array([25., 19., 18.])
    """
    rev = sales * prices # => 20, 5, 0
    return np.sum(rev, axis=1)


def product_revenue(sales: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """2 — Repeat the calculation, change the axis.

    Learn: Arithmetic can broadcast; reductions remove an axis.
    API examples (independent operations):
        grid = np.array([[1, 2], [3, 4]])
        print(grid + np.array([10, 100]))  # [[11, 102], [13, 104]]

        other = np.array([[2, 5, 1], [4, 3, 2]])
        print(other.sum(axis=1))  # [8, 9], shape (2,)
        # axis=1 reduces columns; axis=0 reduces rows.

    Inputs: sales: int ndarray (N,D); prices: float ndarray (D,).
    Output: float ndarray (D,).
    Task: Return each product's total revenue across all days.
    Example input/output:
        sales = np.array([[2, 1, 0], [0, 3, 2], [1, 0, 4]])
        prices = np.array([10., 5., 2.])
        Expected output: np.array([30., 20., 12.])
    """
    rev = sales * prices
    return np.sum(rev, axis=0)


def target_days(day_ids: np.ndarray, sales: np.ndarray, prices: np.ndarray,
                target: float) -> Tuple[np.ndarray, np.ndarray]:
    """3 — Repeat arithmetic, comparison, and aligned masking.

    Learn: Comparisons create boolean arrays; boolean indexing selects.
    API examples (independent operations):
        values = np.array([3, 8, 1])
        print(values < 5)  # [True, False, True]

        letters = np.array(['A', 'B', 'C'])
        keep = np.array([False, True, True])
        print(letters[keep])  # ['B', 'C']
        # A mask on the first axis of a 2D array selects whole rows.

    Inputs: day_ids: int ndarray (N,); sales: int ndarray (N,D);
            prices: float ndarray (D,); target: float.
    Output: tuple (int IDs (M,), int sales (M,D)); M may be zero.
    Task: Return IDs and sales rows for days whose revenue is >= target.
    Example input/output:
        day_ids = np.array([101, 102, 103])
        sales = np.array([[2, 1, 0], [0, 3, 2], [1, 0, 4]])
        prices = np.array([10., 5., 2.])
        target = 19.
        Expected output: (np.array([101, 102]),
                          np.array([[2, 1, 0], [0, 3, 2]]))
    """
    rev_by_day = np.sum(sales * prices, axis=1)
    keep = rev_by_day >= target
    return day_ids[keep], sales[keep]



def sales_shares(sales: np.ndarray) -> np.ndarray:
    """4 — Repeat keepdims, now divide.

    Learn: keepdims preserves a reduced axis with length 1.
    API examples (independent operations):
        grid = np.array([[2, 4, 6], [1, 3, 5]])
        print(grid.max(axis=1))                 # [6, 5], shape (2,)
        print(grid.max(axis=1, keepdims=True))  # [[6], [5]], shape (2, 1)
        # keepdims works with sum and mean too.

        values = np.array([6., 10.])
        print(values / 2.)  # [3., 5.]; / performs elementwise division.

    Inputs: sales: int ndarray (N,D); EVERY row has a positive total.
    Output: float ndarray (N,D).
    Task: Return each product's fraction of the units sold that day.
    Example input/output:
        sales = np.array([[2, 1, 1], [0, 3, 2]])
        Expected output: np.array([[0.5, 0.25, 0.25], [0., 0.6, 0.4]])
    """
    total_sold = sales.sum(axis=1, keepdims=True) # => [[4], [5]]
    return sales / total_sold


def revenue_matmul(sales: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """5 — New: matrix-vector multiplication with @.

    Learn: @ on two 1D arrays computes a dot product: multiply pairs,
    then add them. For a matrix and a vector, it does this for each row:
    (N,D) @ (D,) -> (N,).
    API example:
        a = np.array([2., 3.])
        b = np.array([4., 5.])
        print(a * b)  # [8., 15.]: elementwise products
        print(a @ b)  # 23.0: 2*4 + 3*5, a scalar

    Inputs: sales: int ndarray (N,D); prices: float ndarray (D,).
    Output: float ndarray (N,).
    Task: Return daily revenue using @ rather than multiplication and sum.
    Example input/output:
        sales = np.array([[2, 1, 0], [0, 3, 2], [1, 0, 4]])  # shape (3, 3)
        prices = np.array([10., 5., 2.])  # shape (3,)
        Expected output: np.array([25., 19., 18.])
    """
    return sales @ prices


def predict_revenue(sales: np.ndarray, weights: np.ndarray,
                    bias: float) -> np.ndarray:
    """6 — A first model: X @ w + b.

    Learn: A linear model uses a weighted sum plus a constant bias.
    Weights and bias are given today; later they will be learned.
    API example — scalar broadcasting:
        values = np.array([4., 9., 2.])
        print(values + 0.5)  # [4.5, 9.5, 2.5], same shape as values
        # A scalar is applied to every element, without a loop.

    Inputs: sales: int ndarray (N,D); weights: float ndarray (D,);
            bias: float. Weights and bias may be negative.
    Output: float predictions ndarray (N,); do not clip predictions.
    Task: Return predicted daily revenue using a weighted sum plus bias.
    Example input/output:
        sales = np.array([[2, 1, 0], [0, 3, 2], [1, 0, 4]])
        weights = np.array([10., 5., 2.])
        bias = 3.
        Expected output: np.array([28., 22., 21.])
    """
    return sales @ weights + bias


def forecast_days(day_ids: np.ndarray, sales: np.ndarray,
                  weights: np.ndarray, bias: float,
                  target: float) -> Tuple[np.ndarray, np.ndarray]:
    """7 — Integrated repetition: predict, compare, select.

    Learn: Boolean indexing preserves order and may return no elements.
    API example:
        values = np.array([8., 3., 6.])
        keep = np.array([True, False, True])
        print(values[keep])  # [8., 6.], not sorted
        empty_mask = np.array([False, False, False])
        print(values[empty_mask].shape)  # (0,)

    Inputs: day_ids: int ndarray (N,); sales: int ndarray (N,D);
            weights: float ndarray (D,); bias: float; target: float.
    Output: tuple (int IDs (M,), float predictions (M,)), in input order;
            if no day qualifies, both arrays have shape (0,).
    Task: Return IDs and predictions for days with predicted revenue >= target.
    Example input/output:
        day_ids = np.array([101, 102, 103])
        sales = np.array([[2, 1, 0], [0, 3, 2], [1, 0, 4]])
        weights = np.array([10., 5., 2.])
        bias = 3.
        target = 22.
        Expected output: (np.array([101, 102]), np.array([28., 22.]))
    """
    predicted_rev = sales @ weights + bias
    keep = predicted_rev >= target
    return day_ids[keep], predicted_rev[keep]


def center_sales(sales: np.ndarray) -> np.ndarray:
    """8 — Optional: repeat yesterday's centering (~5 minutes).

    Learn: mean calculates averages; subtraction supports broadcasting.
    API examples (independent operations):
        grid = np.array([[2, 4], [6, 8]])
        print(grid.mean(axis=1, keepdims=True))  # [[3.], [7.]]
        # Choose the axis based on what each row/column represents.

        values = np.array([5., 9., 12.])
        print(values - 2.)  # [3., 7., 10.]

    Inputs: sales: int ndarray (N,D).
    Output: float ndarray (N,D).
    Task: Return sales minus each product's average daily sales.
    Example input/output:
        sales = np.array([[2, 4], [4, 8]])
        Expected output: np.array([[-1., -2.], [1., 2.]])
    """
    avg_daily_sales = sales.mean(axis=0)
    return sales - avg_daily_sales
