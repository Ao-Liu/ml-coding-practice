"""Day 1: NumPy basics through simple broadcasting.

All teaching notes are here: read a function's comments, then implement it.
Exercises 1–5: review your existing work.
Exercises 6–9: slices and boolean masks.
Exercises 10–13: axes and reshaping.
Exercises 14–15: simple broadcasting.
Take breaks between groups; allow roughly 40–60 minutes for the new work.
Run one test at a time with: python -m pytest -q week01_numpy_logreg/day01_numpy -k FUNCTION_NAME
Unfinished functions intentionally raise NotImplementedError.
"""

import numpy as np


def make_array():
    """Exercise 1: return a NumPy array containing 5, 10, 15, 20.

    LEARN: np.array converts a Python list into an ndarray.
    Example: np.array([2, 4, 6]) creates a 1D array with shape (3,).
    TASK: Return an array, not a Python list.
    """
    values = np.array([5, 10, 15, 20])
    return values


def array_shape(x):
    """Exercise 2: return the shape tuple of x."""
    '''
    v = np.array([7, 8, 9])
    print(v.shape)  # (3,) — one dimension with 3 values
    
    m = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print(m.shape)  # (2, 3) — 2 rows, 3 columns
    
    .shape -> (how many rows there are, how many items in each of rows)
    '''
    return x.shape


def second_column(x):
    """Exercise 3: return the second column of a 2D array as a 1D array."""
    '''
    m = np.array([[10, 20, 30],
              [40, 50, 60]])
    print(m[0, 2])  # 30: first row, third column
    print(m[0, :]) # [10 20 30]: all columns in the first row
    print(m[:, 0]) # [10 40]: all rows in the first column
    '''
    return x[:, 1]


def add_ten(x):
    """Exercise 4: add 10 to each value without modifying x."""
    '''
    v = np.array([2, 4, 6])
    print(v + 3) # [5 7 9]
    print(v * 2) # [4 8 12]
    '''
    return x + 10


def row_sums(x):
    """Exercise 5: return one sum per row of a 2D array."""
    '''
    axis=0: collapse the row dimension, giving one result per column.
    axis=1: collapse the column dimension, giving one result per row.
    
    m = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    print(np.sum(m))         # 21
    print(np.sum(m, axis=0)) # [9 12]: one total for each column
    '''
    return np.sum(x, axis=1)


# Continue below, one function at a time. All explanations are in this file.
# Run one exercise from the repository root, for example:
# python -m pytest -q week01_numpy_logreg/day01_numpy -k first_rows
# Use NumPy operations, without loops. Do not modify the inputs.
# Inputs have valid shapes; no input validation is needed today.


def first_rows(x, n):
    """Exercise 6 — Slicing: select the first n rows.

    LEARN: x[start:stop] includes start but excludes stop. Omitting start
    means start at 0. For a 2D array, the first index selects rows.
    A colon in the second position keeps all columns.

    Example:
        a = np.array([[10, 11], [20, 21], [30, 31], [40, 41]])
        a[1:3, :]  # [[20, 21], [30, 31]], shape (2, 2)

    TASK: Return the first n rows of a 2D array, keeping all columns.
    Assume 1 <= n <= the number of rows. A slice/view is fine.
    For x=[[1, 2], [3, 4], [5, 6]], n=2 -> [[1, 2], [3, 4]].
    Before coding: what shape should your result have?
    """
    return x[:n, :]


def every_other(x):
    """Exercise 7 — Slicing variation: use a step.

    LEARN: x[start:stop:step] lets you skip elements. An omitted stop
    means continue to the end.

    Example:
        a = np.array([10, 20, 30, 40, 50, 60])
        a[1::2]  # [20, 40, 60]: start at index 1, advance by 2

    TASK: For a 1D array, return elements at indices 0, 2, 4, ... .
    For x=[7, 8, 9, 10, 11] -> [7, 9, 11]. A slice/view is fine.
    """
    return x[::2]


def positive_values(x):
    """Exercise 8 — Boolean masking: filter values.

    LEARN: Comparing an array with a number produces a boolean array.
    Using that boolean array as an index keeps only the True positions.

    Example:
        a = np.array([2, 7, 4, 9])
        mask = a < 5  # [True, False, True, False]
        a[mask]       # [2, 4]

    TASK: Return only values strictly greater than zero from a 1D array,
    in their original order. Zero is not positive.
    For x=[-3, 0, 2, -1, 5] -> [2, 5].
    If nothing matches, return an empty 1D array.
    """
    mask = x > 0
    return x[mask]


def values_in_range(x, low, high):
    """Exercise 9 — Combine two masks.

    LEARN: Use & for elementwise AND, and put each comparison in
    parentheses. Python's 'and' does not combine NumPy masks.

    Example:
        a = np.array([1, 4, 7, 10])
        mask = (a > 2) & (a < 9)
        a[mask]  # [4, 7]

    TASK: Keep values in the inclusive range [low, high] from a 1D array.
    Include both endpoints and preserve the order. Assume low <= high.
    For x=[1, 3, 5, 7], low=3, high=5 -> [3, 5].
    """
    mask = (x >= low) & (x <= high)
    return x[mask]


def column_means(x):
    """Exercise 10 — Review axis, now with mean.

    LEARN: np.mean computes the average. For shape (N, D):
      axis=0 collapses the rows -> shape (D,), one result per column.
      axis=1 collapses the columns -> shape (N,), one result per row.

    Example:
        a = np.array([[2, 4], [6, 8], [10, 12]])
        np.mean(a, axis=1)  # [3., 7., 11.]: row averages

    TASK: Return one mean per column of a nonempty 2D array.
    For x=[[1, 4, 7], [3, 6, 9]] -> [2., 5., 8.], shape (3,).

    axis = 1 -> num of results = num of rows
    axis = 0 -> num of results = num of cols
    """
    return np.mean(x, axis=0)



def row_maxima(x):
    """Exercise 11 — Aggregation variation: maximum.

    LEARN: np.max uses the same axis rules as sum and mean.
    Without axis, it returns the single largest value in the whole array.

    Example:
        a = np.array([[1, 9], [7, 3]])
        np.max(a)          # 9
        np.max(a, axis=0)  # [7, 9]: column maxima

    TASK: Return one maximum per row of a nonempty 2D array.
    For x=[[1, 8, 2], [9, 3, 4]] -> [8, 9], shape (2,).
    """
    return np.max(x, axis=1)


def reshape_pairs(x):
    """Exercise 12 — Reshape: organize a sequence into a table.

    LEARN: reshape changes the shape without changing the number of
    elements. By default, values fill each row before moving to the next.
    One dimension may be -1: NumPy infers it from the element count.

    Example:
        a = np.array([1, 2, 3, 4, 5, 6])
        a.reshape(2, 3)   # [[1, 2, 3], [4, 5, 6]]
        a.reshape(-1, 3)  # same result: 6 / 3 = 2 rows
        # reshape(4, 2) would fail: it needs 8 values, but a has 6.

    TASK: Reshape a nonempty 1D array into rows of TWO values.
    Its length is even. For x=[3, 6, 9, 12] -> [[3, 6], [9, 12]].
    A view is fine. What is the output shape if x has 10 elements?

    .reshape(x rows, y cols)
    -1:自动计算应该有多少个
    """
    return x.reshape(-1, 2)


def as_column(x):
    """Exercise 13 — Reshape variation: make a column.

    LEARN: (N,) and (N, 1) are different shapes! The first is 1D;
    the second is a 2D table with N rows and one column.

    Example:
        a = np.array([2, 4, 6])
        a.shape            # (3,)
        a.reshape(1, -1)   # [[2, 4, 6]], shape (1, 3): one ROW

    TASK: Turn a nonempty 1D array into a 2D column, shape (N, 1).
    For x=[5, 10, 15] -> [[5], [10], [15]]. A view is fine.
    Check the shape, not just the printed values.
    """
    return x.reshape(-1, 1)


def add_column_offsets(x, offsets):
    """Exercise 14 — Simple broadcasting: one offset per column.

    LEARN: NumPy can combine arrays of different shapes. Compare shapes
    from the right: dimensions must match, or one must be 1. Missing
    leading dimensions act like 1. NumPy reuses values as needed.

    Example:
        a = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
        scale = np.array([2, 10, 100])         # shape    (3,)
        a * scale  # [[2, 20, 300], [8, 50, 600]]
        # Each row uses the same three scale values.

    TASK: Add offsets[j] to every value in column j.
    x has shape (N, D); offsets has shape (D,). Return shape (N, D).
    For x=[[1, 2], [3, 4]], offsets=[10, 100]
    -> [[11, 102], [13, 104]]. Leave both inputs unchanged.

    shape 更准确的意思是：每个维度有多长
    [[1, 2, 3], [4, 5, 6]] 一维是2，二维是3
    [2, 10, 100] 一维是3
    NumPy 从最右边对齐维度，两个 3 正好匹配
    """
    return x + offsets


def scale_rows(x, factors):
    """Exercise 15 — Broadcasting variation: one factor per row.

    LEARN: For shape (N, D), a shape (N,) array does NOT line up with
    the rows: broadcasting aligns dimensions from the right. A column
    of shape (N, 1) does line up and repeats across the D columns.
    Reshaping a 1D array into a column connects this to exercise 13.

    Example:
        a = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        offsets = np.array([[10], [100]])     # (2, 1)
        a + offsets  # [[11, 12, 13], [104, 105, 106]]

    TASK: Multiply every value in row i by factors[i].
    x has shape (N, D); factors is initially 1D with shape (N,).
    Return shape (N, D), leaving inputs unchanged.
    For x=[[1, 2, 3], [4, 5, 6]] (2, 3), factors=[2, 10] (2,)
    -> [[2, 4, 6], [40, 50, 60]].
    Before coding: write the two shapes you want to multiply.
    """
    factors = factors.reshape(-1, 1)
    return x * factors
