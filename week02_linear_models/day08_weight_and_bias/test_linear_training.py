"""Explicit small cases plus one observable learning check."""

import numpy as np
import pytest
import linear_training as t


def check(actual, expected):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.shape == expected.shape
        assert np.issubdtype(actual.dtype, np.floating)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-7)
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for a, e in zip(actual, expected):
            check(a, e)
    else:
        assert type(actual) is float, 'Return a Python float for scalar results.'
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize('name,args,expected', [
    ('compare_weights', ([0,2],1.,0.,[1.,5.],1.), (13.,5.,1.)),
    ('compare_weights', ([0,2],3.,1.,[1.,5.],1.), (0.,2.,8.)),
    ('compare_weights', ([2],1.,1.,[4.],.5), (4.,1.,0.)),
    ('weight_slope', ([0,2],1.,0.,[1.,5.]), -6.),
    ('weight_slope', ([1,3],2.,1.,[5.,6.]), 1.),
    ('weight_slope', ([0,0],2.,1.,[4.,5.]), 0.),
    ('update_weight_once', ([0,2],1.,0.,[1.,5.],.1), (1.6,5.,2.12)),
    ('update_weight_once', ([1,3],2.,1.,[5.,6.],.1), (1.9,2.5,2.45)),
    ('update_weight_once', ([0,0],2.,1.,[4.,5.],.1), (2.,12.5,12.5)),
    ('update_both_once', ([0,2],1.,0.,[1.,5.],.1), (1.6,.4,5.,1.16)),
    ('update_both_once', ([1,3],2.,1.,[5.,6.],.1), (1.9,1.1,2.5,2.32)),
    ('update_both_once', ([0,2],2.,1.,[1.,5.],.1), (2.,1.,0.,0.)),
    ('fit_line', ([0,2],1.,0.,[1.,5.],.1,2), (1.88,.6,np.array([5.,1.16,.2848]))),
    ('fit_line', ([0,2],1.,0.,[1.,5.],.1,0), (1.,0.,np.array([5.]))),
    ('fit_line', ([1,3],2.,1.,[5.,6.],.1,1), (1.9,1.1,np.array([2.5,2.32]))),
    ('fit_line', ([0,2],2.,1.,[1.,5.],.1,3), (2.,1.,np.array([0.,0.,0.,0.]))),
], ids=lambda value: value if isinstance(value, str) else None)
def test_exercise(name, args, expected):
    inputs = tuple(np.array(a) if isinstance(a, list) else a for a in args)
    originals = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in inputs)
    result = getattr(t, name)(*inputs)
    check(result, expected)
    for current, original in zip(inputs, originals):
        np.testing.assert_array_equal(current, original)


def test_fit_line_learns_both_parameters():
    units = np.array([0,2])
    actual = np.array([1.,5.])
    weight, bias, history = t.fit_line(units, 1., 0., actual, .1, 200)
    assert history.shape == (201,)
    assert np.isfinite(history).all()
    assert weight == pytest.approx(2., abs=1e-5)
    assert bias == pytest.approx(1., abs=1e-5)
    assert history[-1] < 1e-10
    np.testing.assert_array_equal(units, [0,2])
    np.testing.assert_array_equal(actual, [1.,5.])
