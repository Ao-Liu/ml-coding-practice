"""Day 10 — Connect yesterday's separate columns to vector operations.

Five exercises, about 35–45 minutes. No new model or loss today.
Rows of sales are days (N); columns are products (D). Instead of separate
weight_a and weight_b scalars, weights is a 1D float array of length D.
actual and errors have shape (N,); bias is still a scalar float.

Start with prediction, then look at transpose alone, then combine it with
a supplied error vector. Only after that do we calculate complete gradients.
Exercise 5 repeats one parameter update. No training loop needed today.

Assume finite valid inputs, N,D >= 1, nonnegative integer sales, float
weights/actual/errors, and positive learning_rate. Do not modify inputs.
Return Python floats for scalar results and NumPy arrays for array results.
Prediction, gradient, and updated-weight arrays must have floating dtype.
Write operations directly in each function; don't call earlier exercises.
Use NumPy without loops over days or products. MSE = mean(error squared).

From the repository root:
  python -m pytest -q week02_linear_models/day10_vector_gradients -k predict_vector
Unfinished functions intentionally raise NotImplementedError.
"""

from typing import Tuple
import numpy as np


def predict_vector(sales: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """1 — Pack the weights, keep the same prediction (~6 minutes).

    Learn:
    Yesterday you added A_units*weight_a and B_units*weight_b, then bias.
    Now weights[0] belongs to column 0, weights[1] to column 1, and so on.
    The order must match the product columns. A third product simply adds
    another column and another weight; the meaning of prediction is unchanged.

    @ between a matrix and a vector takes a dot product for EACH ROW.
    One row describes one day, so it gives one total per day:
    sales (N,D) @ weights (D,) gives (N,). Add the daily bias once.
    The 1D vector is (D,), not a 2D row with shape (1,D).

    Independent API reminder — dot product of two 1D vectors:
        a = np.array([2., 3.])
        b = np.array([4., 1.])
        print(a * b)  # [8., 3.], separate products
        print(a @ b)  # 11.0, their sum

    Task: Return daily predictions using @ and the shared bias.
    Example input/output:
        sales = np.array([[1, 2], [3, 0], [0, 4]])
        weights = np.array([2., 1.])
        bias = 1.
        Expected output: np.array([5., 7., 5.])

    Pause: why are there three predictions but only two weights?
    """
    '''
    sales = np.array([[1, 2], [3, 0], [0, 4]])
        weights = np.array([2., 1.])
    =>
    1 x 2 + 2 x 1 + 1 => 5
    2 x 3 + 1 x 0 + 1 => 7
    ...
    '''
    return sales @ weights + bias


def product_rows(sales: np.ndarray) -> np.ndarray:
    """2 — Look at transpose before doing any gradient math (~5 minutes).

    Learn:
    In sales, one column contains a product's sales on all N days.
    Transpose moves each original column into a row. The result has one
    row per product and one column per day: (N,D) becomes (D,N).
    Entry [day, product] becomes [product, day]; values retain their meaning.
    This is not sorting and not simply reshaping a sequence of values.

    Independent API example:
        grid = np.array([[7, 8, 9], [4, 5, 6]])
        print(grid.T)  # [[7, 4], [8, 5], [9, 6]]
        print(grid.T.shape)  # (3, 2)
    In a rectangular array the changed orientation is easy to see.
    Returning a view is fine here; do not change its values.

    Task: Return a table whose rows contain each product's sales across days.
    Example input/output:
        sales = np.array([[1, 2], [3, 0], [0, 4]])
        Expected output: np.array([[1, 3, 0], [2, 0, 4]])

    Pause: point to product B's sales on the third day in both tables.
    """
    return sales.T


def product_error_sums(sales: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """3 — One sum per product, with errors already provided (~8 minutes).

    Learn:
    Yesterday each weight used its own sales column multiplied by the SAME
    daily error vector. We will first calculate only the sums of these
    products: no prediction, averaging, or factor of 2 in this exercise.

    For an imaginary product with units [2,1,0] over three days and errors
    [e1,e2,e3], its sum is 2*e1 + 1*e2 + 0*e3. Another product uses its
    own units but the same e1,e2,e3: these are errors of TOTAL revenue.
    We need one result per product, not one per day.

    Exercise 2 puts each product into a row. @ then takes each row's dot
    product with the supplied daily errors. Shapes: (D,N) @ (N,) -> (D,).
    This is why transpose helps: it lines up DAYS inside each dot product.
    The operation gathers contributions over days for each product.

    Independent API reminder:
        print(np.array([2., 1., 0.]) @ np.array([3., -1., 4.]))  # 5.0

    Task: Return each product's sum of daily sales times signed error using .T and @.
    Example input/output:
        sales = np.array([[1, 2], [3, 0], [0, 4]])
        errors = np.array([1., -2., 3.])
        Expected output: np.array([-5., 14.])

    Pause: expand each result as three multiplications and two additions.
    """
    return sales.T @ errors


def model_gradients(sales: np.ndarray, weights: np.ndarray, bias: float,
                    actual: np.ndarray) -> Tuple[np.ndarray, float]:
    """4 — Turn those sums into yesterday's gradients (~8 minutes).

    Learn:
    First calculate the complete model's predictions and signed errors.
    For product j, yesterday's rule was 2*mean(column_j * error).
    A mean is a sum divided by N, the number of DAYS. Exercise 3 computes
    all those sums together, so multiply them by 2/N to get all gradients.

    In mathematical notation: dw = (2/N) X^T error.
    X is sales, X^T is its transpose, and dw has one entry per product.
    There is no new derivative here: each entry equals the separate
    single-column calculation you wrote yesterday. Do not divide by D.
    Bias still has db = 2*mean(error), because it affects every day equally.
    Both gradients use the same errors. Return (dw, db), not new parameters.

    Independent API reminders:
        grid = np.array([[1, 2], [3, 4], [5, 6]])
        print(grid.shape[0])  # 3: rows
        print(np.array([3., 6.]) / 3.)  # [1., 2.]

    Task: Return all weight gradients as an array and the bias gradient as a float.
    Example input/output:
        sales = np.array([[1, 2], [3, 0], [0, 4]])
        weights = np.array([2., 1.])
        bias = 1.
        actual = np.array([4., 9., 2.])
        Expected output: (np.array([-3.3333333333333335, 9.333333333333334]),
                          1.3333333333333333)

    Pause: which axis counts days, and which counts the weights to update?
    """
    pred = sales @ weights + bias
    signed_err = pred - actual
    n_days = sales.shape[0]
    dw = (2 / n_days) * (sales.T @ signed_err)
    db = 2 * np.mean(signed_err)
    return dw, float(db)


def vector_step(sales: np.ndarray, weights: np.ndarray, bias: float,
                actual: np.ndarray, learning_rate: float
                ) -> Tuple[np.ndarray, float, float, float]:
    """5 — Update a whole vector of weights once (~10 minutes).

    Learn:
    Repeat the same prediction, errors, and gradients from exercise 4:
    dw = (2/N) X^T error; db = 2*mean(error).
    All gradients must describe the OLD model. Once they are ready,
    subtract learning_rate times each gradient from its matching parameter.
    weights and dw both have shape (D,), so elementwise arithmetic updates
    all products at once, without separate weight_a/weight_b variables.

    With NumPy arrays, avoid weights -= ... here: it would change the
    caller's array. Use an expression that produces a new array instead.
    Update bias too, then recompute predictions to measure the new loss.
    Keep MSE as mean(error squared); a large learning rate can increase it.

    Independent API example — a separate result preserves the input:
        values = np.array([5., 8.])
        offsets = np.array([1., 2.])
        shifted = values - offsets
        print(values)   # [5., 8.]
        print(shifted)  # [4., 6.]

    Task: Return new weights, new bias, old MSE, and new MSE after one joint update.
    Example input/output:
        sales = np.array([[0, 0], [1, 0], [0, 2]])
        weights = np.array([1., 1.])
        bias = 0.
        actual = np.array([1., 3., 7.])
        learning_rate = 0.15
        Expected output: (np.array([1.2, 2.0]), 0.8, 10.0, 1.96)

    Pause: this is Day 9's update result, now packaged as a weight vector.
    The same function should work for one, two, or three products.
    """
    pred = sales @ weights + bias
    signed_err = pred - actual
    n_days = sales.shape[0]
    dw = (2 / n_days) * (sales.T @ signed_err)
    db = 2 * np.mean(signed_err)
    old_mse = np.mean(signed_err ** 2)

    weights = weights - (learning_rate * dw)
    bias -= learning_rate * db
    pred = sales @ weights + bias
    signed_err = pred - actual
    return weights, float(bias), float(old_mse), float(np.mean(signed_err ** 2))
