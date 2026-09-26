"""Day 14 — Use standardized features in a familiar training loop (~40 minutes).

Scenario: predict daily revenue from two products' sales. Yesterday you learned
how to choose a ruler for each column. Today keep that ruler through training
and prediction. Revenue targets stay in their original units.

Work in order. Repeat the NumPy operations yourself in 1–4; exercise 5 may reuse
these functions to keep the final integration small. All arrays contain finite
numbers, training columns have positive population std, and datasets are nonempty.
No constant-column handling, new derivatives, or early stopping today.
"""
from typing import Tuple, List
import numpy as np


def training_stats(train_sales: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """1 — Review: choose the rulers from training days (~5 minutes).

    Learn:
    Rows are days and columns are products. For a table (N,D), reducing axis=0
    combines days and leaves one number per product, shape (D,). Compute the
    mean and population std separately for each column; use ddof=0.
    Do NOT take std of the column means: that would compare different products
    instead of measuring how each product changes across days.
    For a column [1,3], center=2 and std=1; [10,30] has center=20 and std=10.
    These two pairs describe different units but the same relative spread.

    Independent API reminder:
        grid = np.array([[2., 7.], [4., 9.], [6., 11.]])
        print(grid.sum(axis=0))  # [12., 27.], shape (2,)
        print(np.std(np.array([5., 9.]), ddof=0))  # 2.0

    Task: Return a tuple of training column means and population standard deviations.
    Example input/output:
        train_sales = np.array([[1,10], [3,10], [1,30], [3,30]])
        Expected output: (np.array([2.,20.]), np.array([1.,10.]))

    Pause: why are there two returned statistics per product, not per day?
    """
    col_means = train_sales.mean(axis=0)
    std = np.std(train_sales, axis=0, ddof=0)
    return col_means, std



def transform_features(sales: np.ndarray, means: np.ndarray,
                       scales: np.ndarray) -> np.ndarray:
    """2 — Review: use the supplied rulers, including on new days (~5 minutes).

    Learn:
    Subtract each column's mean, then divide by that column's std. The supplied
    means/scales both have shape (D,); broadcasting applies them to every row
    of sales (N,D). All supplied scales are positive. Return a float table.
    A new value 40 with training center 20 and std 10 becomes +2. It stays +2
    even if every new day has value 40. Do not calculate new statistics here.
    A negative transformed value simply means below the training average;
    it is valid input to a linear model, not a negative count of sold items.

    Independent API reminder:
        grid = np.array([[3., 8.], [5., 12.]])
        print(grid / np.array([1., 4.]))  # [[3.,2.], [5.,3.]]

    Task: Return a new standardized table using the supplied means and scales.
    Example input/output:
        sales = np.array([[2,20], [4,40]])
        means = np.array([2.,20.])
        scales = np.array([1.,10.])
        Expected output: np.array([[0.,0.], [2.,2.]])

    Pause: should a single new row automatically become all zeros?
    """
    return (sales - means) / scales


def fit_scaled_model(features: np.ndarray, weights: np.ndarray, bias: float,
                     actual: np.ndarray, learning_rate: float, steps: int
                     ) -> Tuple[np.ndarray, float, List[float]]:
    """3 — Repeat your training loop on already standardized features (~10 minutes).

    Learn:
    features is ALREADY standardized, shape (N,D). No preprocessing is needed
    inside this function. Predictions still use features @ weights + bias.
    Signed errors still mean prediction minus actual revenue, shape (N,).
    Standardizing inputs changes their numbers, not the training procedure.

    Recall dw = (2/N) * features.T @ errors and db = 2 * mean(errors).
    Each weight gradient combines its column with the same daily errors; N is
    the number of days, not products. Compute both gradients before changing
    either parameter, then subtract learning_rate times each gradient.
    Recompute predictions with the current parameters on every iteration.
    Keep the initial MSE followed by MSE after each update: steps+1 values.
    Copy weights first so even steps=0 returns an independent array. Do not
    modify any input arrays. Bias and history entries are Python floats.

    Independent API reminders:
        original = np.array([3., 7.])
        working = original.copy()
        working[0] = 0.
        print(original)  # [3.,7.]
        history = [8.0]
        history.append(5.0)
        print(history)  # [8.0,5.0]

    Task: Return weights, bias, and MSE history after exactly steps updates.
    Example input/output:
        features = np.array([[-1.,-1.], [1.,-1.], [-1.,1.], [1.,1.]])
        weights = np.array([0.,0.])
        bias = 0.
        actual = np.array([0.,2.,4.,6.])
        learning_rate = 0.1
        steps = 2
        Expected output: (np.array([0.36,0.72]), 1.08, [14.,8.96,5.7344])

    Pause: which parts of your earlier training loop changed? Only the input values.
    """
    weights = weights.copy()
    pred = features @ weights + bias
    signed_err = pred - actual
    n = features.shape[0]
    mse_history = [np.mean(signed_err ** 2)]

    for _ in range(steps):
        dw = (2 / n) * (features.T @ signed_err)
        db = 2 * np.mean(signed_err)
        weights = weights - (dw * learning_rate)
        bias -= (db * learning_rate)
        pred = features @ weights + bias
        signed_err = pred - actual
        mse_history.append(np.mean(signed_err ** 2))

    return weights, float(bias), mse_history

def predict_raw_days(sales: np.ndarray, means: np.ndarray, scales: np.ndarray,
                     weights: np.ndarray, bias: float) -> np.ndarray:
    """4 — Connect the model to new raw sales (~7 minutes).

    Learn:
    These weights were learned on STANDARDIZED columns. Before predicting,
    transform raw sales with the saved TRAINING means and scales. Then use the
    familiar matrix-vector prediction. Do not train or estimate statistics.
    For one product, center=20 and std=10 map raw sales 40 to 2. A learned
    weight 5 contributes 2*5=10 to revenue, not 40*5=200. The weight expects
    the same input units it saw during training.
    Revenue was never standardized, so predictions already have revenue units;
    do not multiply predictions by feature scales or add feature means.
    Return a float array (N,), even when there is only one new day.

    Independent API reminder:
        grid = np.array([[2., 0.], [1., 3.]])
        print(grid @ np.array([4., 1.]))  # [8.,7.]

    Task: Return predictions for raw sales using the supplied training statistics and fitted parameters.
    Example input/output:
        sales = np.array([[2,20], [4,40]])
        means = np.array([2.,20.])
        scales = np.array([1.,10.])
        weights = np.array([0.36,0.72])
        bias = 1.08
        Expected output: np.array([1.08,3.24])

    Pause: why would sales @ weights + bias directly give the wrong predictions?
    """
    standardized = (sales - means) / scales
    return standardized @ weights + bias


def train_scaled_and_validate(train_sales: np.ndarray, train_actual: np.ndarray,
                              validation_sales: np.ndarray, validation_actual: np.ndarray,
                              weights: np.ndarray, bias: float,
                              learning_rate: float, steps: int
                              ) -> Tuple[np.ndarray, float, float, float]:
    """5 — Put the familiar pieces together (~12 minutes).

    Learn:
    Choose means and scales using only train_sales, and use them for BOTH sets.
    Train using only standardized training rows and train_actual. After the
    final update, evaluate that fixed model on training and validation rows.
    Both MSE values compare revenue predictions with their own actual targets.
    Never standardize targets. Validation data must not affect the rulers,
    gradients, or fitted parameters; it only affects validation MSE.

    You may reuse exercises 1–4 here: the practice is arranging the stages.
    Keep this order in mind: choose rulers -> transform -> fit -> evaluate.
    The two sets may have different row counts but the same product columns.
    Validation can have a constant column or just one day; use training scales
    anyway. Return an independent weight array, including when steps=0; both
    MSE values and bias are Python floats. Leave all input arrays unchanged.

    Independent API reminder — unpacking a tuple:
        summary = (np.array([2., 5.]), 3.0)
        values, offset = summary
        print(values.shape, offset)  # (2,) 3.0

    Task: Return fitted weights, fitted bias, final training MSE, and validation MSE.
    Example input/output:
        train_sales = np.array([[1,10], [3,10], [1,30], [3,30]])
        train_actual = np.array([0.,2.,4.,6.])
        validation_sales = np.array([[2,20], [4,40]])
        validation_actual = np.array([3.,9.])
        weights = np.array([0.,0.])
        bias = 0.
        learning_rate = 0.1
        steps = 2
        Expected output: (np.array([0.36,0.72]), 1.08, 5.7344, 18.432)

    Pause: if only validation targets change, which three outputs stay fixed?
    """
    weights = weights.copy()
    train_mean = np.mean(train_sales, axis=0)
    train_scale = np.std(train_sales, axis=0, ddof=0)
    train_stand = (train_sales - train_mean) / train_scale

    train_pred = train_stand @ weights + bias
    signed_err = train_pred - train_actual
    n = train_sales.shape[0]

    for _ in range(steps):
        dw = (2 / n) * (train_stand.T @ signed_err)
        db = 2 * np.mean(signed_err)
        weights = weights - (dw * learning_rate)
        bias -= (db * learning_rate)
        train_pred = train_stand @ weights + bias
        signed_err = train_pred - train_actual

    train_mse = np.mean(signed_err ** 2)
    validation_stand = (validation_sales - train_mean) / train_scale
    v_pred = validation_stand @ weights + bias
    v_signed_err = v_pred - validation_actual
    v_mse = np.mean(v_signed_err ** 2)
    return weights, float(bias), float(train_mse), float(v_mse)
