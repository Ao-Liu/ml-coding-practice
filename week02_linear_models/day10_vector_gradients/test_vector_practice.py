"""Rectangular shapes, separate products, and preservation of input arrays."""

import numpy as np
import pytest
import vector_practice as v


def check(actual, expected):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.shape == expected.shape
        kind = np.floating if expected.dtype.kind == 'f' else np.integer
        assert np.issubdtype(actual.dtype, kind)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for a, e in zip(actual, expected):
            check(a, e)
    else:
        assert type(actual) is float
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize('name,args,expected', [
    ('predict_vector', ([[1,2],[3,0],[0,4]], [2.,1.],1.), np.array([5.,7.,5.])),
    ('predict_vector', ([[1,2,0],[0,1,3]], [1.,0.,1.],1.), np.array([2.,4.])),
    ('predict_vector', ([[2]], [-1.],.5), np.array([-1.5])),
    ('product_rows', ([[1,2],[3,0],[0,4]],), np.array([[1,3,0],[2,0,4]])),
    ('product_rows', ([[7,8,9]],), np.array([[7],[8],[9]])),
    ('product_rows', ([[2],[5]],), np.array([[2,5]])),
    ('product_error_sums', ([[1,2],[3,0],[0,4]], [1.,-2.,3.]), np.array([-5.,14.])),
    ('product_error_sums', ([[1,2,0],[0,1,3]], [1.,-2.]), np.array([1.,0.,-6.])),
    ('product_error_sums', ([[0,2],[0,1]], [0.,0.]), np.array([0.,0.])),
    ('model_gradients', ([[1,2],[3,0],[0,4]], [2.,1.],1.,[4.,9.,2.]), (np.array([-10/3,28/3]),4/3)),
    ('model_gradients', ([[1,2,0],[0,1,3]], [1.,0.,1.],1.,[1.,6.]), (np.array([1.,0.,-6.]),-1.)),
    ('model_gradients', ([[1],[3]], [2.],1.,[5.,6.]), (np.array([1.]),-1.)),
    ('vector_step', ([[0,0],[1,0],[0,2]], [1.,1.],0.,[1.,3.,7.],.15),
     (np.array([1.2,2.]),.8,10.,1.96)),
    ('vector_step', ([[1,2,0],[0,1,3]], [1.,0.,1.],1.,[1.,6.],.1),
     (np.array([.9,0.,1.6]),1.1,2.5,.505)),
    ('vector_step', ([[1],[3]], [2.],1.,[5.,6.],.1), (np.array([1.9]),1.1,2.5,2.32)),
    ('vector_step', ([[0,0],[1,0],[0,2]], [2.,3.],1.,[1.,3.,7.],.15),
     (np.array([2.,3.]),1.,0.,0.)),
], ids=lambda value: value if isinstance(value, str) else None)
def test_exercise(name, args, expected):
    inputs = tuple(np.array(a) if isinstance(a, list) else a for a in args)
    originals = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in inputs)
    check(getattr(v, name)(*inputs), expected)
    for current, original in zip(inputs, originals):
        np.testing.assert_array_equal(current, original)
