"""Small tests for one concept at a time; no solution implementations."""

import numpy as np
import pytest

from exercises import make_array, array_shape, second_column, add_ten, row_sums


def assert_array(actual, expected):
    assert isinstance(actual, np.ndarray), "Return a NumPy array, not a list."
    np.testing.assert_array_equal(actual, expected)


def test_make_array():
    assert_array(make_array(), np.array([5, 10, 15, 20]))


@pytest.mark.parametrize("x, expected", [
    (np.array([4, 5, 6, 7]), (4,)),
    (np.array([[8, 9], [6, 7], [4, 5]]), (3, 2)),
])
def test_array_shape(x, expected):
    assert array_shape(x) == expected


@pytest.mark.parametrize("x, expected", [
    (np.array([[1, 2, 3], [4, 5, 6]]), np.array([2, 5])),
    (np.array([[8, 9]]), np.array([9])),
])
def test_second_column(x, expected):
    assert_array(second_column(x), expected)


@pytest.mark.parametrize("x, expected", [
    (np.array([0, -2, 5]), np.array([10, 8, 15])),
    (np.array([[1., 2.5], [-10., 0.]]), np.array([[11., 12.5], [0., 10.]])),
])
def test_add_ten(x, expected):
    original = x.copy()
    assert_array(add_ten(x), expected)
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("x, expected", [
    (np.array([[2, 3, 4], [10, 0, 1]]), np.array([9, 11])),
    (np.array([[5, -2]]), np.array([3])),
])
def test_row_sums(x, expected):
    assert_array(row_sums(x), expected)

from exercises import (
    first_rows, every_other, positive_values, values_in_range,
    column_means, row_maxima, reshape_pairs, as_column,
    add_column_offsets, scale_rows,
)


@pytest.mark.parametrize("function, args, expected", [
    pytest.param(first_rows, ([[1, 2], [3, 4], [5, 6]], 2), [[1, 2], [3, 4]], id="first_rows"),
    pytest.param(first_rows, ([[1, 2], [3, 4]], 1), [[1, 2]], id="first_rows_single"),
    pytest.param(every_other, ([7, 8, 9, 10, 11],), [7, 9, 11], id="every_other"),
    pytest.param(every_other, ([2, 4, 6, 8],), [2, 6], id="every_other_even"),
    pytest.param(positive_values, ([-3, 0, 2, -1, 5],), [2, 5], id="positive_values"),
    pytest.param(positive_values, ([-2, 0],), [], id="positive_values_empty"),
    pytest.param(values_in_range, ([1, 3, 5, 7], 3, 5), [3, 5], id="values_in_range"),
    pytest.param(values_in_range, ([5, 2, 5, 0], 5, 5), [5, 5], id="values_in_range_equal"),
    pytest.param(column_means, ([[1, 4, 7], [3, 6, 9]],), [2., 5., 8.], id="column_means"),
    pytest.param(column_means, ([[1, 2], [2, 5], [4, 6]],), [7/3, 13/3], id="column_means_fractional"),
    pytest.param(row_maxima, ([[1, 8, 2], [9, 3, 4]],), [8, 9], id="row_maxima"),
    pytest.param(row_maxima, ([[-5, -2, -9]],), [-2], id="row_maxima_negative"),
    pytest.param(reshape_pairs, ([3, 6, 9, 12],), [[3, 6], [9, 12]], id="reshape_pairs"),
    pytest.param(reshape_pairs, ([1, 2, 3, 4, 5, 6],), [[1, 2], [3, 4], [5, 6]], id="reshape_pairs_longer"),
    pytest.param(as_column, ([5, 10, 15],), [[5], [10], [15]], id="as_column"),
    pytest.param(as_column, ([7],), [[7]], id="as_column_single"),
    pytest.param(add_column_offsets, ([[1, 2], [3, 4]], [10, 100]), [[11, 102], [13, 104]], id="add_column_offsets"),
    pytest.param(add_column_offsets, ([[1, 2, 3], [4, 5, 6]], [0, -2, 10]), [[1, 0, 13], [4, 3, 16]], id="add_column_offsets_rectangular"),
    pytest.param(scale_rows, ([[1, 2, 3], [4, 5, 6]], [2, 10]), [[2, 4, 6], [40, 50, 60]], id="scale_rows"),
    pytest.param(scale_rows, ([[1, 2], [3, 4], [5, 6]], [0, -1, 0.5]), [[0, 0], [-3, -4], [2.5, 3]], id="scale_rows_rectangular"),
])
def test_new_exercise(function, args, expected):
    inputs = tuple(np.array(a) if isinstance(a, list) else a for a in args)
    originals = [a.copy() if isinstance(a, np.ndarray) else a for a in inputs]
    actual = function(*inputs)
    expected = np.array(expected)
    assert isinstance(actual, np.ndarray), "Return a NumPy array, not a list."
    assert actual.shape == expected.shape, "Check the dimensions of your result."
    np.testing.assert_allclose(actual, expected)
    for actual_input, original in zip(inputs, originals):
        np.testing.assert_array_equal(actual_input, original)
