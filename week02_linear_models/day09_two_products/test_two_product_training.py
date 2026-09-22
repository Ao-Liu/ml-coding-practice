"""Independent numeric expectations for two explicit product columns."""

import numpy as np
import pytest
import two_product_training as t


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
        assert type(actual) is float, 'Return Python floats for scalar results.'
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize('name,args,expected', [
    ('predict_two_products', ([[0,0],[1,0],[0,2]],1.,1.,0.), np.array([0.,1.,2.])),
    ('predict_two_products', ([[0,1],[2,0],[1,3]],1.,2.,.5), np.array([2.5,2.5,7.5])),
    ('predict_two_products', ([[2,3]],-1.,.5,2.), np.array([1.5])),
    ('change_a_weight', ([[0,0],[1,0],[0,2]],1.,1.,0.,.5),
     (np.array([0.,1.,2.]),np.array([0.,1.5,2.]))),
    ('change_a_weight', ([[0,1],[2,0],[1,3]],1.,2.,.5,.5),
     (np.array([2.5,2.5,7.5]),np.array([2.5,3.5,8.]))),
    ('separate_gradients', ([[0,0],[1,0],[0,2]],1.,1.,0.,[1.,3.,7.]), (-4/3,-20/3,-16/3)),
    ('separate_gradients', ([[1,0],[0,2],[1,1]],1.,1.,0.,[0.,6.,1.]), (4/3,-14/3,-4/3)),
    ('separate_gradients', ([[0,0],[0,0]],2.,3.,1.,[4.,5.]), (0.,0.,-7.)),
    ('separate_gradients', ([[2,3]],1.,2.,1.,[10.]), (-4.,-6.,-2.)),
    ('update_three_parameters', ([[0,0],[1,0],[0,2]],1.,1.,0.,[1.,3.,7.],.15),
     (1.2,2.,.8,10.,1.96)),
    ('update_three_parameters', ([[1,0],[0,2],[1,1]],1.,1.,0.,[0.,6.,1.],.15),
     (.8,1.7,.2,6.,9.65/3)),
    ('update_three_parameters', ([[0,0],[1,0],[0,2]],2.,3.,1.,[1.,3.,7.],.15),
     (2.,3.,1.,0.,0.)),
    ('fit_two_products', ([[0,0],[1,0],[0,2]],1.,1.,0.,[1.,3.,7.],.15,2),
     (1.3,2.44,1.14,np.array([10.,1.96,.4312]))),
    ('fit_two_products', ([[0,0],[1,0],[0,2]],1.,1.,0.,[1.,3.,7.],.15,0),
     (1.,1.,0.,np.array([10.]))),
    ('fit_two_products', ([[1,0],[0,2],[1,1]],1.,1.,0.,[0.,6.,1.],.15,1),
     (.8,1.7,.2,np.array([6.,9.65/3]))),
    ('fit_two_products', ([[0,0],[1,0],[0,2]],2.,3.,1.,[1.,3.,7.],.15,3),
     (2.,3.,1.,np.array([0.,0.,0.,0.]))),
], ids=lambda value: value if isinstance(value, str) else None)
def test_exercise(name, args, expected):
    inputs = tuple(np.array(a) if isinstance(a, list) else a for a in args)
    originals = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in inputs)
    check(getattr(t, name)(*inputs), expected)
    for current, original in zip(inputs, originals):
        np.testing.assert_array_equal(current, original)
