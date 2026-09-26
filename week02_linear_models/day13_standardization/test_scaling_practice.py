"""Column direction, population std, broadcasting, and training-only statistics."""

import numpy as np
import pytest
import scaling_practice as s


def check(actual, expected):
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for a, e in zip(actual, expected):
            check(a, e)
    else:
        assert isinstance(actual, np.ndarray)
        expected = np.asarray(expected, dtype=float)
        assert actual.shape == expected.shape
        assert np.issubdtype(actual.dtype, np.floating)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-7)


@pytest.mark.parametrize('name,args,expected', [
    ('feature_means', ([[1,100],[3,300]],), [2.,200.]),
    ('feature_means', ([[1,10],[2,20],[6,30]],), [3.,20.]),
    ('center_features', ([[1,100],[3,300]],), [[-1.,-100.],[1.,100.]]),
    ('center_features', ([[1,10],[2,20],[6,30]],), [[-2.,-10.],[-1.,0.],[3.,10.]]),
    ('feature_scales', ([[1,100],[3,300]],), [1.,100.]),
    ('feature_scales', ([[1,10],[2,20],[3,30]],), [.816496580927726,8.16496580927726]),
    ('apply_standardization', ([[1,100],[3,300]],[2.,200.],[1.,100.]), [[-1.,-1.],[1.,1.]]),
    ('apply_standardization', ([[4,500],[2,200],[0,100]],[2.,200.],[1.,100.]), [[2.,3.],[0.,0.],[-2.,-1.]]),
    ('apply_standardization', ([[8]],[2.],[3.]), [[2.]]),
    ('prepare_features', ([[1,100],[3,300]],[[4,500],[2,200],[0,100]]),
     (np.array([[-1.,-1.],[1.,1.]]),np.array([[2.,3.],[0.,0.],[-2.,-1.]]),np.array([2.,200.]),np.array([1.,100.]))),
    ('prepare_features', ([[1],[3]],[[4],[4]]),
     (np.array([[-1.],[1.]]),np.array([[2.],[2.]]),np.array([2.]),np.array([1.]))),
], ids=lambda value: value if isinstance(value, str) else None)
def test_exercise(name, args, expected):
    inputs = tuple(np.array(a) for a in args)
    originals = tuple(a.copy() for a in inputs)
    with np.errstate(divide='raise', invalid='raise'):
        check(getattr(s, name)(*inputs), expected)
    for current, original in zip(inputs, originals):
        np.testing.assert_array_equal(current, original)


def test_prepare_features_validation_does_not_change_training_statistics():
    train = np.array([[1,100],[3,300]])
    first = s.prepare_features(train, np.array([[4,500]]))
    second = s.prepare_features(train, np.array([[100,10000],[200,20000]]))
    check(first[0], second[0])
    check(first[2], second[2])
    check(first[3], second[3])
    check(second[1], [[98.,98.],[198.,198.]])
