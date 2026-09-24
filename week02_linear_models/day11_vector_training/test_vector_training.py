"""Fresh-gradient, shape, history, and ownership checks."""

import numpy as np
import pytest
import vector_training as t


def check(actual, expected):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.shape == expected.shape
        kind = np.floating if expected.dtype.kind == 'f' else np.integer
        assert np.issubdtype(actual.dtype, kind)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-7)
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for a, e in zip(actual, expected):
            check(a, e)
    else:
        assert type(actual) is float
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize('name,args,expected', [
    ('one_step', ([[0,0],[1,0],[0,2]],[1.,1.],0.,[1.,3.,7.],.15), (np.array([1.2,2.]),.8,1.96)),
    ('one_step', ([[1,2,0],[0,1,3]],[1.,0.,1.],1.,[1.,6.],.1), (np.array([.9,0.,1.6]),1.1,.505)),
    ('two_steps', ([[0,0],[1,0],[0,2]],[1.,1.],0.,[1.,3.,7.],.15), (np.array([1.3,2.44]),1.14,.4312)),
    ('two_steps', ([[0],[2]],[1.],0.,[1.,5.],.1), (np.array([1.88]),.6,.2848)),
    ('fit_parameters', ([[0,0],[1,0],[0,2]],[1.,1.],0.,[1.,3.,7.],.15,2), (np.array([1.3,2.44]),1.14)),
    ('fit_parameters', ([[1,2,0],[0,1,3]],[1.,0.,1.],1.,[1.,6.],.1,0), (np.array([1.,0.,1.]),1.)),
    ('fit_parameters', ([[1,2,0],[0,1,3]],[1.,0.,1.],1.,[1.,6.],.1,1), (np.array([.9,0.,1.6]),1.1)),
    ('training_history', ([[0,0],[1,0],[0,2]],[1.,1.],0.,[1.,3.,7.],.15,2),
     (np.array([1.3,2.44]),1.14,np.array([10.,1.96,.4312]))),
    ('training_history', ([[0],[2]],[1.],0.,[1.,5.],.1,2), (np.array([1.88]),.6,np.array([5.,1.16,.2848]))),
    ('training_history', ([[1,2,0],[0,1,3]],[1.,0.,1.],1.,[1.,6.],.1,0), (np.array([1.,0.,1.]),1.,np.array([2.5]))),
    ('training_history', ([[0,0],[1,0],[0,2]],[2.,3.],1.,[1.,3.,7.],.15,3),
     (np.array([2.,3.]),1.,np.array([0.,0.,0.,0.]))),
    ('inspect_errors', ([101,102,103],[[0,0],[1,0],[0,2]],[1.3,2.44],1.14,[1.,3.,7.],.5),
     (.4312,np.array([102,103]),np.array([-.56,-.98]))),
    ('inspect_errors', ([8,3],[[1],[2]],[2.],0.,[5.,1.],1.), (9.,np.array([8,3]),np.array([-3.,3.]))),
    ('inspect_errors', ([8,3],[[1],[2]],[2.],0.,[5.,1.],3.),
     (9.,np.empty(0,dtype=int),np.empty(0,dtype=float))),
], ids=lambda value: value if isinstance(value, str) else None)
def test_exercise(name, args, expected):
    inputs = tuple(np.array(a) if isinstance(a, list) else a for a in args)
    originals = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in inputs)
    result = getattr(t, name)(*inputs)
    check(result, expected)
    for current, original in zip(inputs, originals):
        np.testing.assert_array_equal(current, original)
    if name in ('fit_parameters', 'training_history'):
        assert not np.shares_memory(result[0], inputs[1]), 'Return independent weights, even for zero steps.'
