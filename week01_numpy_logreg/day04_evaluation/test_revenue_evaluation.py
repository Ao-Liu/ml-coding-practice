"""Result, shape, and boundary checks; no solution implementations."""

import numpy as np
import pytest
import revenue_evaluation as e


def check(actual, expected):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.shape == expected.shape
        kind = np.floating if expected.dtype.kind == 'f' else np.integer
        assert np.issubdtype(actual.dtype, kind)
        np.testing.assert_allclose(actual, expected)
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            check(a, b)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys()
        for key in expected:
            check(actual[key], expected[key])
    elif isinstance(expected, float):
        assert type(actual) is float, 'Return a Python float.'
        assert actual == pytest.approx(expected)
    elif isinstance(expected, int):
        assert type(actual) is int, 'Return a Python int ID.'
        assert actual == expected
    else:
        assert actual == expected


@pytest.mark.parametrize('name,args,expected', [
    ('predict_revenue', ([[2,1],[0,3],[1,2]], [4.,2.], 1.), np.array([11.,7.,9.])),
    ('predict_revenue', ([[1,2,3]], [-2.,.5,1.], -4.), np.array([-2.])),
    ('prediction_errors', ([11.,7.,9.], [10.,9.,6.]), np.array([1.,-2.,3.])),
    ('prediction_errors', ([2.5], [2.5]), np.array([0.])),
    ('mean_absolute_error', ([11.,7.,9.], [10.,9.,6.]), 2.),
    ('mean_absolute_error', ([8.,12.], [10.,10.]), 2.),
    ('mean_absolute_error', ([1.5], [1.5]), 0.),
    ('mean_squared_error', ([11.,7.,9.], [10.,9.,6.]), 14/3),
    ('mean_squared_error', ([8.,12.], [10.,10.]), 4.),
    ('mean_squared_error', ([1.5], [1.5]), 0.),
    ('inaccurate_days', ([101,102,103], [[2,1],[0,3],[1,2]], [4.,2.], 1., [10.,9.,6.], 1.),
     (np.array([102,103]), np.array([-2.,3.]))),
    ('inaccurate_days', ([9,3], [[1],[2]], [2.], 0., [3.,3.], 1.),
     (np.empty(0,dtype=int), np.empty(0,dtype=float))),
    ('inaccurate_days', ([9,3], [[1],[2]], [2.], 0., [5.,1.], 0.),
     (np.array([9,3]), np.array([-3.,3.]))),
    ('compare_models', ([[2,1],[0,3],[1,2]], [10.,9.,6.], [4.,2.], 1., [3.,3.], 0.), ('B',14/3,10/3)),
    ('compare_models', ([[1],[2]], [3.,5.], [2.], 0., [2.], 0.), ('A',1.,1.)),
    ('compare_models', ([[1],[2]], [3.,5.], [2.], 1., [1.], 0.), ('A',0.,6.5)),
    ('compare_models', ([[1],[2]], [3.,5.], [1.], 0., [2.], 1.), ('B',6.5,0.)),
    ('evaluation_report', ([101,102,103], [[2,1],[0,3],[1,2]], [4.,2.], 1., [10.,9.,6.]),
     {'predictions':np.array([11.,7.,9.]), 'mse':14/3, 'worst_day_id':103}),
    ('evaluation_report', ([80,20], [[1],[2]], [2.], 0., [7.,3.]),
     {'predictions':np.array([2.,4.]), 'mse':13., 'worst_day_id':80}),
    ('evaluation_report', ([80,20], [[1],[2]], [2.], 0., [4.,2.]),
     {'predictions':np.array([2.,4.]), 'mse':4., 'worst_day_id':80}),
    ('evaluation_report', ([17], [[2,3]], [1.,2.], 1., [9.]),
     {'predictions':np.array([9.]), 'mse':0., 'worst_day_id':17}),
], ids=lambda value: value if isinstance(value, str) else None)
def test_exercise(name, args, expected):
    inputs = tuple(np.array(a) if isinstance(a, list) else a for a in args)
    originals = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in inputs)
    result = getattr(e, name)(*inputs)
    check(result, expected)
    for actual_input, original in zip(inputs, originals):
        np.testing.assert_array_equal(actual_input, original)
