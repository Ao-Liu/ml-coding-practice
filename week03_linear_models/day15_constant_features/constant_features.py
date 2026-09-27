"""Day 15 — A product with the same sales every training day (~40 minutes).

Keep yesterday's sales model. One product now has constant training sales.
The only new rule: replace an exactly zero training std with a scale of 1.
All data are finite, tables are nonempty (N,D), and column order is consistent.
Use ddof=0. Do not change input arrays. No near-zero tolerance is needed today.
Repeat NumPy directly in 1–4; exercise 5 may reuse safe_training_stats.
"""
from typing import Tuple
import numpy as np


def constant_products(train_sales: np.ndarray) -> np.ndarray:
    """1 — Review column std and boolean masks (~5 minutes).

    Learn:
    A column [7,7,7] has mean 7. Every distance from that mean is zero,
    so its standard deviation is zero. This column did not vary in training.
    Compute std across days (axis=0), leaving shape (D,), then compare with 0.
    The comparison returns one boolean per product; return that whole mask,
    not the selected sales values. Nonzero constant columns count too.

    Independent API reminder:
        values = np.array([2., 5., 2.])
        print(values == 2.)  # [True, False, True]
        print(np.std(np.array([4., 8.]), ddof=0))  # 2.0

    Task: Return a boolean mask marking products with zero training standard deviation.
    Example input/output:
        train_sales = np.array([[1,7,0], [3,7,0], [5,7,0]])
        Expected output: np.array([False, True, True])

    Pause: is a column constant only when all its values are zero?
    """
    std = np.std(train_sales, axis=0, ddof=0)
    mask = std == 0
    return mask


def safe_training_stats(train_sales: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """2 — New: choose a usable scale for a constant column (~8 minutes).

    Learn:
    Standardization is (value - mean) / scale. With a constant training column,
    both the centered value and std are zero, and 0/0 is undefined.
    Keep the real mean but replace each exactly zero std with 1 BEFORE dividing.
    Positive std values stay unchanged. Return means and safe scales, both (D,).
    This is a preprocessing convention: the actual std remains zero.
    For [7,7], mean=7 and safe scale=1; each training value becomes (7-7)/1=0.
    Adding 1 to EVERY scale would also change ordinary columns, so use a mask
    to replace only zero entries. There is no need for a loop over products.

    Independent API example — assignment at selected positions:
        values = np.array([3., -2., 8.])
        copied = values.copy()
        copied[copied < 0.] = 0.
        print(copied)  # [3.,0.,8.]
        print(values)  # [3.,-2.,8.]

    Task: Return training means and scales with zero standard deviations replaced by 1.
    Example input/output:
        train_sales = np.array([[1,7,0], [3,7,0]])
        Expected output: (np.array([2.,7.,0.]), np.array([1.,1.,1.]))

    Pause: why must replacing the scale happen before division?
    """
    train_means = np.mean(train_sales, axis=0)
    train_scales = np.std(train_sales, axis=0, ddof=0)
    train_scales[train_scales == 0] = 1
    return train_means, train_scales


def prepare_safe_features(train_sales: np.ndarray, validation_sales: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """3 — Repeat training-only preprocessing with the new rule (~8 minutes).

    Learn:
    Estimate means and std using train_sales only. Replace zero std with 1,
    then transform both tables with those same means and safe scales.
    Keep every product column, including constant ones, in the same order.
    Return training features, validation features, means, and safe scales.

    A column constant during training can change on a new day. With training
    mean 7 and safe scale 1, a validation value 9 becomes (9-7)/1=2, not zero.
    Do not force that entire validation column to zero or estimate its own std.
    For this special column the scale is our chosen 1, not a measured std of 1.
    Different row counts are fine: both tables only need the same D columns.

    Independent API reminder — broadcast subtraction:
        grid = np.array([[3.,9.], [5.,12.]])
        print(grid - np.array([1.,4.]))  # [[2.,5.], [4.,8.]]

    Task: Return standardized training and validation tables, training means, and safe scales.
    Example input/output:
        train_sales = np.array([[1,7], [3,7]])
        validation_sales = np.array([[4,9], [2,7], [0,6]])
        Expected output: (np.array([[-1.,0.], [1.,0.]]),
                          np.array([[2.,2.], [0.,0.], [-2.,-1.]]),
                          np.array([2.,7.]), np.array([1.,1.]))

    Pause: which outputs can change if only validation_sales changes?
    """
    train_mean = np.mean(train_sales, axis=0)
    train_scale = np.std(train_sales, axis=0, ddof=0)
    train_scale[train_scale == 0] = 1
    train_stand = (train_sales - train_mean) / train_scale

    validation_stand = (validation_sales - train_mean) / train_scale
    return train_stand, validation_stand, train_mean, train_scale


def predict_with_stats(sales: np.ndarray, means: np.ndarray, safe_scales: np.ndarray,
                       weights: np.ndarray, bias: float) -> np.ndarray:
    """4 — Review prediction with saved statistics (~6 minutes).

    Learn:
    The saved scales are already positive, including any replacement 1s.
    Standardize sales using the supplied arrays, then calculate predictions
    with the weights learned on standardized features. Return shape (N,).
    No statistics or parameters are estimated here. Targets and predictions
    remain in revenue units. Do not 'undo' standardization on predictions.
    A training-constant product keeps its column position, so its weight still
    lines up with the right column. Dropping that column would change shapes.

    Independent API reminder:
        grid = np.array([[2.,0.], [0.,3.]])
        print(grid @ np.array([4.,5.]) + 1.)  # [9.,16.]

    Task: Return revenue predictions using the supplied training means, safe scales, and parameters.
    Example input/output:
        sales = np.array([[4,9], [2,7], [0,6]])
        means = np.array([2.,7.])
        safe_scales = np.array([1.,1.])
        weights = np.array([0.72,0.])
        bias = 1.08
        Expected output: np.array([2.52,1.08,-0.36])

    Pause: does a single prediction row give you enough reason to recompute scales?
    """
    train_standardized = (sales - means) / safe_scales
    return train_standardized @ weights + bias


def fit_safe_model(train_sales: np.ndarray, actual: np.ndarray,
                   weights: np.ndarray, bias: float, learning_rate: float, steps: int
                   ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """5 — Repeat fitting and keep the rulers for future predictions (~12 minutes).

    Learn:
    Compute training means and safe scales once, standardize the training table,
    then run yesterday's loop exactly steps times. You may reuse exercise 2
    for the statistics; write the training loop yourself for repetition.
    Return fitted weights, bias, training means, and safe scales, in that order.
    These four values are everything exercise 4 needs alongside new sales.

    Recall prediction = X @ weights + bias, where X is the standardized table.
    errors = prediction - actual, dw = (2/N) * (X.T @ errors),
    and db = 2 * mean(errors). N counts DAYS. Both gradients use the same current
    errors. Subtract learning_rate times each gradient, then recompute errors
    for the next iteration. Do not standardize actual revenue.

    A constant training column becomes all zeros. Its contribution to its own
    gradient is therefore zero on every day. That weight stays at its INITIAL
    value: zero if initialized at zero, but not automatically zero otherwise.
    Training cannot learn an effect for a feature that never varies here.
    Other weights and bias still update normally. This adds no gradient rule.
    Copy weights before updating; even steps=0 must return independent weights.
    With zero steps still compute and return the training statistics.

    Independent API reminder:
        original = np.array([2.,6.])
        working = original.copy()
        working -= 1.
        print(original)  # [2.,6.]
        print(working)   # [1.,5.]

    Task: Return fitted weights, bias as a Python float, training means, and safe scales.
    Example input/output:
        train_sales = np.array([[1,7], [3,7]])
        actual = np.array([1.,5.])
        weights = np.array([0.,0.])
        bias = 0.
        learning_rate = 0.1
        steps = 2
        Expected output: (np.array([0.72,0.]), 1.08,
                          np.array([2.,7.]), np.array([1.,1.]))

    Pause: if the second weight started at 4 instead, would this training change it?
    """
    weights = weights.copy()
    train_means = np.mean(train_sales, axis=0)
    train_scales = np.std(train_sales, axis=0, ddof=0)
    train_scales[train_scales == 0] = 1
    train_standardized = (train_sales - train_means) / train_scales

    pred = train_standardized @ weights + bias
    signed_err = pred - actual
    n = train_standardized.shape[0]

    for _ in range(steps):
        dw = (2 / n) * (train_standardized.T @ signed_err)
        db = 2 * np.mean(signed_err)
        weights = weights - (dw * learning_rate)
        bias -= (db * learning_rate)
        pred = train_standardized @ weights + bias
        signed_err = pred - actual

    return weights, float(bias), train_means, train_scales
