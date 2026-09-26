"""Day 13 — Put feature columns on comparable scales (35–45 minutes).

Same shop: rows are days, columns are products. One product may sell a few
units while another sells hundreds. Such scale differences can make gradient
descent harder to tune. Today prepare features; do not train a model yet.
Standardization can help optimization, but does not guarantee better accuracy.

Five steps: column means → centering → standard deviations → transform with
provided statistics → learn statistics on TRAINING rows only.

All inputs are finite 2D NumPy arrays (integer or float). No input validation
is required. Arrays are nonempty and product columns have consistent order.
For standard deviations computed today, every column varies (std > 0).
Supplied scales are positive. Constant columns and zero division come later.
Do not modify inputs. Return floating-point NumPy arrays. Use NumPy without
loops and repeat operations directly rather than calling earlier exercises.

A negative standardized value means below the training mean, not negative
physical sales. We transform features only; actual revenues stay in their
original units. All code examples teach separate operations, not solutions.

From the repository root:
  python -m pytest -q week02_linear_models/day13_standardization -k feature_means
Unfinished functions intentionally raise NotImplementedError.
"""

from typing import Tuple
import numpy as np


def feature_means(sales: np.ndarray) -> np.ndarray:
    """1 — Review: each product needs its own average (~5 minutes).

    Learn:
    A feature is one input column, here a product's daily sales. Its mean
    tells us its typical value across the observed days. We need one mean
    per product, not one mean per day and not a single mean for the table.
    For (N,D), reducing days leaves (D,). Think which axis represents days.

    Independent API reminder — the OTHER direction, averaging each row:
        grid = np.array([[2, 4], [6, 10], [8, 12]])
        print(grid.mean(axis=1))  # [3., 8., 10.], shape (3,)
    axis=0 reduces rows; axis=1 reduces columns. Neither axis is 'always
    correct': choose based on what one row and one column mean.

    Task: Return one mean per product as a 1D array.
    Example input/output:
        sales = np.array([[1, 100], [3, 300]])
        Expected output: np.array([2., 200.])

    Pause: why should the output length equal the number of columns?
    """
    return sales.mean(axis=0)


def center_features(sales: np.ndarray) -> np.ndarray:
    """2 — Review: measure how far each value is from its column mean (~6 minutes).

    Learn:
    Centering subtracts a column's mean from each value in that column.
    A value above the mean becomes positive; one below becomes negative.
    For a feature with mean 10, a value of 13 becomes +3 and 8 becomes -2.
    Centering changes the reference point, but the spread is unchanged.

    Broadcasting lets one mean per column apply to every row:
    (N,D) combined with (D,) still produces (N,D). After centering, each
    column has mean approximately zero, allowing for floating-point error.
    You have done this before; try remembering the operations first.

    Independent API example — broadcasting addition:
        grid = np.array([[1, 2], [3, 4], [5, 6]])
        print(grid + np.array([10, 100]))
        # [[11, 102], [13, 104], [15, 106]]

    Task: Return sales with each column's mean subtracted from that column.
    Example input/output:
        sales = np.array([[1, 100], [3, 300]])
        Expected output: np.array([[-1., -100.], [1., 100.]])

    Pause: the means are now zero, but are the column spreads comparable yet?
    """
    col_mean = sales.mean(axis=0)
    return sales - col_mean


def feature_scales(sales: np.ndarray) -> np.ndarray:
    """3 — New: standard deviation measures spread (~8 minutes).

    Learn:
    The mean describes a center; standard deviation describes spread
    around it, in the SAME units as the original values. It is nonnegative.
    The population definition is sqrt(mean((value - mean)^2)).
    NumPy's np.std calculates this directly; use ddof=0 today.

    A small numeric example: [6,10] has mean 8, deviations [-2,2], squared
    deviations [4,4], mean square 4, and standard deviation sqrt(4)=2.
    We square so opposite deviations do not cancel, then take a square root
    to return to the original units. You need not implement these steps;
    they explain what std means. Apply it separately to each product column.

    Independent API example — a 1D array:
        values = np.array([2., 2., 6., 6.])
        print(np.std(values, ddof=0))  # 2.0
    Like mean, std accepts an axis argument for a 2D table. Every column
    in this exercise varies, so its standard deviation is positive.

    Task: Return one population standard deviation per product as a 1D array.
    Example input/output:
        sales = np.array([[1, 100], [3, 300]])
        Expected output: np.array([1., 100.])

    Pause: which product varies more in absolute units, and by what factor?
    """
    return np.std(sales, axis=0, ddof=0)


def apply_standardization(sales: np.ndarray, means: np.ndarray,
                          scales: np.ndarray) -> np.ndarray:
    """4 — Center, then express differences in standard-deviation units (~8 minutes).

    Learn:
    Standardization uses z = (value - mean) / standard_deviation.
    A value 6 above its mean with std=3 becomes +2: two standard deviations
    above the mean. A value 3 below that same mean becomes -1.
    Centering alone keeps original units; dividing by std removes that scale.
    Standardized values are NOT probabilities and are not bounded to [0,1].

    means and scales are supplied 1D arrays, each length D. Use them as
    given; do not recompute statistics from sales. Broadcasting applies
    each column's own mean and scale across all its rows. All scales > 0.
    This function can transform training rows or completely new rows.

    Independent API reminder — elementwise division:
        print(np.array([6., 10.]) / np.array([2., 5.]))  # [3., 2.]
    Operation order matters: subtract the center before dividing.

    Task: Return sales standardized with the supplied means and scales.
    Example input/output:
        sales = np.array([[1, 100], [3, 300]])
        means = np.array([2., 200.])
        scales = np.array([1., 100.])
        Expected output: np.array([[-1., -1.], [1., 1.]])

    Pause: would these means/scales necessarily make NEW rows have mean zero?
    """
    return (sales - means) / scales


def prepare_features(train_sales: np.ndarray, validation_sales: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """5 — Use training statistics on both sets (~10 minutes).

    Learn:
    Yesterday validation rows were excluded from gradient updates. Today
    exclude them from estimating preprocessing statistics too. Calculate
    each column's mean and population std using TRAINING rows only, then
    use those same means/scales to transform both tables.

    Think of choosing one ruler from training data. Measuring validation
    with a different ruler would change the meaning of a model's inputs.
    For example, training mean 10 and std 2 map a new value 14 to +2, even
    if 14 happens to be the average of the validation set. Do not recenter
    validation separately. Its transformed columns need not average zero.
    Computing statistics on combined data would let held-out feature values
    influence preprocessing; today's evaluation keeps them out entirely.

    The two tables can have different numbers of days but share D columns
    in the same product order. All training columns have positive std;
    validation columns may be constant. Transform only features, not targets.
    Later a model must use these same training statistics for future inputs.

    Independent API reminder — reductions can preserve a dimension:
        grid = np.array([[2, 4], [6, 8]])
        print(grid.sum(axis=1, keepdims=True))  # [[6], [14]]
    Here return means and scales as 1D arrays (D,), without an extra axis.

    Task: Return standardized training data, standardized validation data, training means, and training scales.
    Example input/output:
        train_sales = np.array([[1, 100], [3, 300]])
        validation_sales = np.array([[4, 500], [2, 200], [0, 100]])
        Expected output: (np.array([[-1., -1.], [1., 1.]]),
                          np.array([[2., 3.], [0., 0.], [-2., -1.]]),
                          np.array([2., 200.]), np.array([1., 100.]))

    Pause: if only validation_sales changes, which returned values stay fixed?
    """
    train_mean = train_sales.mean(axis=0)
    train_scale = np.std(train_sales, axis=0, ddof=0)
    train_standardized = (train_sales - train_mean) / train_scale

    validation_standardized = (validation_sales - train_mean) / train_scale

    return train_standardized, validation_standardized, train_mean, train_scale
