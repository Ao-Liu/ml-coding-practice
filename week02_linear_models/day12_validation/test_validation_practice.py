"""Aligned splits, fixed-model evaluation, and training-only updates."""

import numpy as np
import pytest
import validation_practice as v


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
    ('split_days', ([[0,0],[1,0],[0,2],[1,1],[2,0]], [1.,3.,7.,6.,5.],3),
     (np.array([[0,0],[1,0],[0,2]]),np.array([1.,3.,7.]),np.array([[1,1],[2,0]]),np.array([6.,5.]))),
    ('split_days', ([[2],[4],[6]], [3.,5.,7.],1),
     (np.array([[2]]),np.array([3.]),np.array([[4],[6]]),np.array([5.,7.]))),
    ('predict_later_days', ([[1,1],[2,0]],[2.,3.],1.),np.array([6.,5.])),
    ('predict_later_days', ([[1,2,3]],[1.,0.,-1.],.5),np.array([-1.5])),
    ('validation_score', ([[1,1],[2,0]],[6.,5.],[1.3,2.44],1.14), (np.array([4.88,3.74]),1.421)),
    ('validation_score', ([[1,1],[2,0]],[6.,5.],[2.,3.],1.), (np.array([6.,5.]),0.)),
    ('train_and_validate', ([[0,0],[1,0],[0,2]],[1.,3.,7.],[[1,1],[2,0]],[6.,5.],[1.,1.],0.,.15,2),
     (np.array([1.3,2.44]),1.14,.4312,1.421)),
    ('train_and_validate', ([[0,0],[1,0],[0,2]],[1.,3.,7.],[[1,1],[2,0]],[6.,5.],[1.,1.],0.,.15,0),
     (np.array([1.,1.]),0.,10.,12.5)),
    ('train_and_validate', ([[0],[2]],[1.,5.],[[1]],[3.],[1.],0.,.1,2),
     (np.array([1.88]),.6,.2848,.2704)),
    ('difficult_validation_days', ([501,502],[[1,1],[2,0]],[6.,5.],[1.3,2.44],1.14,1.2),
     (np.array([502]),np.array([-1.26]))),
    ('difficult_validation_days', ([8,3],[[1],[2]],[5.,1.],[2.],0.,1.),
     (np.array([8,3]),np.array([-3.,3.]))),
    ('difficult_validation_days', ([8,3],[[1],[2]],[5.,1.],[2.],0.,3.),
     (np.empty(0,dtype=int),np.empty(0,dtype=float))),
], ids=lambda value: value if isinstance(value, str) else None)
def test_exercise(name, args, expected):
    inputs = tuple(np.array(a) if isinstance(a, list) else a for a in args)
    originals = tuple(a.copy() if isinstance(a, np.ndarray) else a for a in inputs)
    result = getattr(v, name)(*inputs)
    check(result, expected)
    for current, original in zip(inputs, originals):
        np.testing.assert_array_equal(current, original)
    if name == 'train_and_validate':
        assert not np.shares_memory(result[0], inputs[4])


def test_train_and_validate_validation_does_not_train_model():
    train_x = np.array([[0,0],[1,0],[0,2]])
    train_y = np.array([1.,3.,7.])
    valid_x = np.array([[1,1],[2,0]])
    initial = np.array([1.,1.])
    first = v.train_and_validate(train_x, train_y, valid_x, np.array([6.,5.]), initial, 0., .15, 2)
    second = v.train_and_validate(train_x, train_y, valid_x, np.array([0.,0.]), initial, 0., .15, 2)
    check(first, (np.array([1.3,2.44]),1.14,.4312,1.421))
    check(second, (np.array([1.3,2.44]),1.14,.4312,18.901))
