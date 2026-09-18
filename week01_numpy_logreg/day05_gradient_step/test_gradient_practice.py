"""Explicit results for gradient steps, shapes, and input preservation."""

import numpy as np
import pytest
import gradient_practice as g


def check(actual, expected):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.shape == expected.shape
        kind = np.floating if expected.dtype.kind == 'f' else np.integer
        assert np.issubdtype(actual.dtype, kind)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-7)
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            check(a, b)
    else:
        assert type(actual) is float, 'Return a Python float for scalar results.'
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize('name,args,expected', [
    ('prediction_snapshot', ([[1,0],[0,1]], [0.,0.], 0., [2.,4.]),
     (np.array([0.,0.]), np.array([-2.,-4.]), 10.)),
    ('prediction_snapshot', ([[1,2,0],[0,1,3]], [1.,0.,1.], 1., [1.,6.]),
     (np.array([2.,4.]), np.array([1.,-2.]), 2.5)),
    ('bias_gradient', ([[1,0],[0,1]], [0.,0.], 0., [2.,4.]), -6.),
    ('bias_gradient', ([[1],[3]], [2.], 1., [5.,6.]), -1.),
    ('bias_gradient', ([[1],[2]], [1.], 0., [0.,3.]), 0.),
    ('update_bias', ([[1,0],[0,1]], [0.,0.], 0., [2.,4.], .1), (.6,6.76)),
    ('update_bias', ([[1],[3]], [2.], 1., [5.,6.], .5), (1.5,2.25)),
    ('update_bias', ([[1,0],[0,1]], [0.,0.], 0., [2.,4.], 2.), (12.,82.)),
    ('single_weight_gradient', ([1,2], 0., 0., [2.,4.]), -10.),
    ('single_weight_gradient', ([1,3], 2., 1., [5.,6.]), 1.),
    ('single_weight_gradient', ([0,0], 2., 1., [5.,6.]), 0.),
    ('weight_gradients', ([[1,0],[0,1]], [0.,0.], 0., [2.,4.]), np.array([-2.,-4.])),
    ('weight_gradients', ([[1,2,0],[0,1,3]], [1.,0.,1.], 1., [1.,6.]), np.array([1.,0.,-6.])),
    ('weight_gradients', ([[1],[3]], [2.], 1., [5.,6.]), np.array([1.])),
    ('train_step', ([[1,0],[0,1]], [0.,0.], 0., [2.,4.], .1),
     (np.array([.2,.4]), .6, 10., 5.22)),
    ('train_step', ([[1,0],[0,1]], [0.,0.], 0., [2.,4.], 1.),
     (np.array([2.,4.]), 6., 10., 36.)),
    ('train_step', ([[1,2,0],[0,1,3]], [1.,0.,1.], 1., [1.,6.], .1),
     (np.array([.9,0.,1.6]), 1.1, 2.5, .505)),
    ('train_step', ([[2,1]], [1.,2.], 1., [5.], .1),
     (np.array([1.,2.]), 1., 0., 0.)),
    ('remaining_bad_days', ([101,102], [[1,0],[0,1]], [.2,.4], .6, [2.,4.], 2.),
     (np.array([102]), np.array([-3.]))),
    ('remaining_bad_days', ([9,3], [[1],[2]], [2.], 0., [5.,1.], 1.),
     (np.array([9,3]), np.array([-3.,3.]))),
    ('remaining_bad_days', ([9,3], [[1],[2]], [2.], 0., [5.,1.], 3.),
     (np.empty(0,dtype=int), np.empty(0,dtype=float))),
], ids=lambda value: value if isinstance(value, str) else None)
def test_exercise(name, args, expected):
    inputs = tuple(np.array(a) if isinstance(a, list) else a for a in args)
    originals = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in inputs)
    result = getattr(g, name)(*inputs)
    check(result, expected)
    for current, original in zip(inputs, originals):
        np.testing.assert_array_equal(current, original)
