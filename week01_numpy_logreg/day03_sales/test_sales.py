"""Small examples and shape checks; no reference implementations."""

import numpy as np
import pytest
import sales_practice as s


def arr(actual, expected):
    assert isinstance(actual, np.ndarray), 'Return a NumPy array.'
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    expected_kind = np.floating if expected.dtype.kind == 'f' else np.integer
    assert np.issubdtype(actual.dtype, expected_kind)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('name,args,expected', [
    ('daily_revenue', ([[2,1,0],[0,3,2],[1,0,4]], [10.,5.,2.]), [25.,19.,18.]),
    ('daily_revenue', ([[2,0],[0,0]], [1.5,2.]), [3.,0.]),
    ('product_revenue', ([[2,1,0],[0,3,2],[1,0,4]], [10.,5.,2.]), [30.,20.,12.]),
    ('product_revenue', ([[1,2]], [1.5,2.]), [1.5,4.]),
    ('target_days', ([101,102,103], [[2,1,0],[0,3,2],[1,0,4]], [10.,5.,2.], 19.),
     (np.array([101,102]), np.array([[2,1,0],[0,3,2]]))),
    ('target_days', ([7], [[0,0]], [1.,2.], 1.),
     (np.empty(0, dtype=int), np.empty((0,2), dtype=int))),
    ('sales_shares', ([[2,1,1],[0,3,2]],), [[.5,.25,.25],[0.,.6,.4]]),
    ('sales_shares', ([[3],[7]],), [[1.],[1.]]),
    ('revenue_matmul', ([[2,1,0],[0,3,2],[1,0,4]], [10.,5.,2.]), [25.,19.,18.]),
    ('revenue_matmul', ([[2,0],[0,3]], [1.5,2.]), [3.,6.]),
    ('predict_revenue', ([[2,1,0],[0,3,2],[1,0,4]], [10.,5.,2.], 3.), [28.,22.,21.]),
    ('predict_revenue', ([[1,2],[3,0]], [-2.,.5], -1.), [-2.,-7.]),
    ('forecast_days', ([101,102,103], [[2,1,0],[0,3,2],[1,0,4]], [10.,5.,2.], 3.,22.),
     (np.array([101,102]), np.array([28.,22.]))),
    ('forecast_days', ([7], [[1,2]], [1.,2.], 0.,100.),
     (np.empty(0,dtype=int), np.empty(0,dtype=float))),
    ('forecast_days', ([80,20], [[1],[2]], [2.], 1.,3.),
     (np.array([80,20]), np.array([3.,5.]))),
    ('center_sales', ([[2,4],[4,8]],), [[-1.,-2.],[1.,2.]]),
    ('center_sales', ([[1,2,3]],), [[0.,0.,0.]]),
], ids=lambda value: value if isinstance(value, str) else None)
def test_exercise(name, args, expected):
    inputs = tuple(np.array(a) if isinstance(a, list) else a for a in args)
    originals = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in inputs)
    result = getattr(s, name)(*inputs)
    if isinstance(expected, tuple):
        assert isinstance(result, tuple) and len(result) == 2
        for actual_part, expected_part in zip(result, expected):
            arr(actual_part, expected_part)
    else:
        arr(result, expected)
    for actual_input, original in zip(inputs, originals):
        np.testing.assert_array_equal(actual_input, original)
