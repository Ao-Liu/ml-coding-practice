"""Check update counts, fresh gradients, histories, and independent runs."""

import numpy as np
import pytest
import bias_training as b


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
    ('one_update', ([1,3], 2., 0., [5.,9.], .1), (.6,5.76)),
    ('one_update', ([1,3], 2., 4., [3.,7.], .1), (3.4,5.76)),
    ('two_updates', ([1,3], 2., 0., [5.,9.], .1), (1.08,3.6864)),
    ('two_updates', ([1,3], 2., 4., [3.,7.], .1), (2.92,3.6864)),
    ('fit_bias', ([1,3], 2., 0., [5.,9.], .1,3), 1.464),
    ('fit_bias', ([1,3], 2., 4., [3.,7.], .1,0), 4.),
    ('fit_bias', ([1,3], 2., 4., [3.,7.], .5,5), 1.),
    ('loss_history', ([1,3], 2., 0., [5.,9.], .1,3),
     (1.464,np.array([9.,5.76,3.6864,2.359296]))),
    ('loss_history', ([1,3], 2., 4., [3.,7.], .1,0), (4.,np.array([9.]))),
    ('loss_history', ([1,2], 2., 0., [1.,5.], .1,3), (0.,np.array([1.,1.,1.,1.]))),
    ('loss_history', ([1,3], 2., 0., [5.,9.], 1.1,2),
     (-1.32,np.array([9.,12.96,18.6624]))),
    ('compare_learning_rates', ([1,3], 2., 0., [5.,9.], [.1,.5,1.1],2),
     (np.array([1.08,3.,-1.32]),np.array([3.6864,0.,18.6624]))),
    ('compare_learning_rates', ([1,3], 2., 4., [3.,7.], [.5,.1,.5],1),
     (np.array([1.,3.4,1.]),np.array([0.,5.76,0.]))),
    ('compare_learning_rates', ([1,3], 2., 4., [3.,7.], [.1,.5],0),
     (np.array([4.,4.]),np.array([9.,9.]))),
    ('compare_learning_rates', ([2], 1., 0., [4.], [.25],2),
     (np.array([1.5]),np.array([.25]))),
], ids=lambda value: value if isinstance(value, str) else None)
def test_exercise(name, args, expected):
    inputs = tuple(np.array(a) if isinstance(a, list) else a for a in args)
    originals = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in inputs)
    result = getattr(b, name)(*inputs)
    check(result, expected)
    for current, original in zip(inputs, originals):
        np.testing.assert_array_equal(current, original)
