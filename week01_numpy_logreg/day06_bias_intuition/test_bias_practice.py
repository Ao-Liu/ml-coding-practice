"""Small examples, direction changes, and input-preservation checks."""

import numpy as np
import pytest
import bias_practice as b


def check(actual, expected):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.shape == expected.shape
        assert np.issubdtype(actual.dtype, np.floating)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for a, e in zip(actual, expected):
            check(a, e)
    else:
        assert type(actual) is float, 'Return a Python float for scalar results.'
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize('name,args,expected', [
    ('change_bias', ([1,3], 2., 0., 1.), (np.array([2.,6.]), np.array([3.,7.]))),
    ('change_bias', ([0,2,5], 1.5, -1., .5), (np.array([-1.,2.,6.5]), np.array([-.5,2.5,7.]))),
    ('change_weight', ([1,3], 2., 0., 1.), (np.array([2.,6.]), np.array([3.,9.]))),
    ('change_weight', ([0,2,5], 1.5, -1., .5), (np.array([-1.,2.,6.5]), np.array([-1.,3.,9.]))),
    ('compare_biases', ([1,3], 2., 0., [5.,9.], 1.), (16.,9.,4.)),
    ('compare_biases', ([1,3], 2., 4., [3.,7.], 1.), (4.,9.,16.)),
    ('compare_biases', ([1,2], 2., 0., [1.,5.], 1.), (2.,1.,2.)),
    ('compare_biases', ([2], 1., 1., [4.], .5), (2.25,1.,.25)),
    ('bias_slope', ([1,3], 2., 0., [5.,9.]), -6.),
    ('bias_slope', ([1,3], 2., 4., [3.,7.]), 6.),
    ('bias_slope', ([1,2], 2., 0., [1.,5.]), 0.),
    ('bias_slope', ([2], 1., 1., [4.]), -2.),
    ('update_bias_once', ([1,3], 2., 0., [5.,9.], .1), (.6,9.,5.76)),
    ('update_bias_once', ([1,3], 2., 4., [3.,7.], .1), (3.4,9.,5.76)),
    ('update_bias_once', ([1,2], 2., 0., [1.,5.], .1), (0.,1.,1.)),
    ('update_bias_once', ([2], 1., 1., [4.], .25), (1.5,1.,.25)),
    ('update_bias_once', ([1,3], 2., 0., [5.,9.], 2.), (12.,9.,81.)),
], ids=lambda value: value if isinstance(value, str) else None)
def test_exercise(name, args, expected):
    inputs = tuple(np.array(a) if isinstance(a, list) else a for a in args)
    originals = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in inputs)
    result = getattr(b, name)(*inputs)
    check(result, expected)
    for current, original in zip(inputs, originals):
        np.testing.assert_array_equal(current, original)
