"""Safe scales, column alignment, training isolation, and repeated fitting."""
import numpy as np
import pytest
import constant_features as s


def check(actual, expected):
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for a, e in zip(actual, expected):
            check(a, e)
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray) and actual.shape == expected.shape
        if expected.dtype == bool:
            assert actual.dtype == bool
            np.testing.assert_array_equal(actual, expected)
        else:
            assert np.issubdtype(actual.dtype, np.floating)
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, atol=1e-10)
    else:
        assert isinstance(actual, float)
        assert actual == pytest.approx(expected, abs=1e-10)


def call(name, *args):
    before = [a.copy() if isinstance(a, np.ndarray) else None for a in args]
    with np.errstate(divide='raise', invalid='raise'):
        result = getattr(s, name)(*args)
    for arg, old in zip(args, before):
        if old is not None:
            np.testing.assert_array_equal(arg, old)
    return result


@pytest.mark.parametrize('raw,expected', [
    ([[1,7,0],[3,7,0],[5,7,0]], [False,True,True]),
    ([[2,10],[4,30]], [False,False]),
    ([[5,8]], [True,True]),
])
def test_constant_products(raw, expected):
    check(call('constant_products', np.array(raw)), np.array(expected))


@pytest.mark.parametrize('raw,means,scales', [
    ([[1,7,0],[3,7,0]], [2.,7.,0.], [1.,1.,1.]),
    ([[2,9,10],[6,9,30]], [4.,9.,20.], [2.,1.,10.]),
    ([[5,8]], [5.,8.], [1.,1.]),
    ([[0.],[0.0002]], [0.0001], [0.0001]),
])
def test_safe_training_stats(raw, means, scales):
    check(call('safe_training_stats', np.array(raw)), (np.array(means),np.array(scales)))


def test_prepare_safe_features_example():
    check(call('prepare_safe_features', np.array([[1,7],[3,7]]),np.array([[4,9],[2,7],[0,6]])),
          (np.array([[-1.,0.],[1.,0.]]),np.array([[2.,2.],[0.,0.],[-2.,-1.]]),
           np.array([2.,7.]),np.array([1.,1.])))


def test_prepare_safe_features_training_isolation():
    train = np.array([[2,9,10],[6,9,30]])
    a = call('prepare_safe_features', train, np.array([[8,12,40]]))
    b = call('prepare_safe_features', train, np.array([[8,12,40],[4,9,20]]))
    check(a, (np.array([[-1.,0.,-1.],[1.,0.,1.]]),np.array([[2.,3.,2.]]),
              np.array([4.,9.,20.]),np.array([2.,1.,10.])))
    check(b[0],a[0])
    check(b[2:],a[2:])
    check(b[1],np.array([[2.,3.,2.],[0.,0.,0.]]))


def test_prepare_safe_features_all_constant():
    check(call('prepare_safe_features',np.array([[7],[7]]),np.array([[9]])),
          (np.array([[0.],[0.]]),np.array([[2.]]),np.array([7.]),np.array([1.])))


@pytest.mark.parametrize('raw,means,scales,w,b,expected', [
    ([[4,9],[2,7],[0,6]], [2.,7.], [1.,1.], [.72,0.], 1.08, [2.52,1.08,-.36]),
    ([[8,12]], [4.,9.], [2.,1.], [2.,3.], 1., [14.]),
])
def test_predict_with_stats(raw, means, scales, w, b, expected):
    check(call('predict_with_stats',np.array(raw),np.array(means),np.array(scales),np.array(w),b),
          np.array(expected))


@pytest.mark.parametrize('initial_constant', [0.,4.])
def test_fit_safe_model_two_steps(initial_constant):
    w = np.array([0.,initial_constant])
    result = call('fit_safe_model',np.array([[1,7],[3,7]]),np.array([1.,5.]),w,0.,.1,2)
    check(result, (np.array([.72,initial_constant]),1.08,np.array([2.,7.]),np.array([1.,1.])))
    assert not np.shares_memory(result[0],w)


def test_fit_safe_model_zero_steps():
    w = np.array([2.,3.])
    result = call('fit_safe_model',np.array([[2,9],[6,9]]),np.array([1.,5.]),w,1.,.1,0)
    check(result, (np.array([2.,3.]),1.,np.array([4.,9.]),np.array([2.,1.])))
    assert not np.shares_memory(result[0],w)


def test_fit_safe_model_all_constant_bias_still_learns():
    check(call('fit_safe_model',np.array([[7],[7],[7]]),np.array([1.,3.,5.]),np.array([4.]),0.,.1,2),
          (np.array([4.]),1.08,np.array([7.]),np.array([1.])))
