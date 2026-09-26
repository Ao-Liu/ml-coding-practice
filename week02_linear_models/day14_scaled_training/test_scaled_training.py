"""Check shapes, fixed training statistics, repeated updates, and input ownership."""
import numpy as np
import pytest
import scaled_training as s


def check(actual, expected):
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            check(a, e)
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.shape == expected.shape
        assert np.issubdtype(actual.dtype, np.floating)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    else:
        assert isinstance(actual, float)
        assert actual == pytest.approx(expected, abs=1e-10)


def call(name, *args):
    originals = [a.copy() if isinstance(a, np.ndarray) else None for a in args]
    with np.errstate(divide='raise', invalid='raise'):
        result = getattr(s, name)(*args)
    for arg, old in zip(args, originals):
        if old is not None:
            np.testing.assert_array_equal(arg, old)
    return result


def data():
    return (np.array([[1,10],[3,10],[1,30],[3,30]]),
            np.array([0.,2.,4.,6.]), np.array([[2,20],[4,40]]),
            np.array([3.,9.]), np.array([0.,0.]))


@pytest.mark.parametrize('raw,means,scales', [
    ([[1,10],[3,10],[1,30],[3,30]], [2.,20.], [1.,10.]),
    ([[2],[4],[6]], [4.], [np.sqrt(8/3)]),
])
def test_training_stats(raw, means, scales):
    check(call('training_stats', np.array(raw)), (np.array(means), np.array(scales)))


@pytest.mark.parametrize('raw,expected', [
    ([[2,20],[4,40]], [[0.,0.],[2.,2.]]),
    ([[4,40]], [[2.,2.]]),
    ([[4,40],[4,40]], [[2.,2.],[2.,2.]]),
])
def test_transform_features(raw, expected):
    check(call('transform_features', np.array(raw), np.array([2.,20.]),
               np.array([1.,10.])), np.array(expected))


def test_fit_scaled_model_two_steps():
    x = np.array([[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.]])
    check(call('fit_scaled_model', x, np.zeros(2), 0., np.array([0.,2.,4.,6.]), .1, 2),
          (np.array([.36,.72]), 1.08, [14.,8.96,5.7344]))


def test_fit_scaled_model_zero_steps():
    w = np.array([1.])
    result = call('fit_scaled_model', np.array([[-1.],[1.]]), w, 2., np.array([0.,4.]), .1, 0)
    check(result, (np.array([1.]), 2., [1.]))
    assert not np.shares_memory(result[0], w)


@pytest.mark.parametrize('raw,expected', [
    ([[2,20],[4,40]], [1.08,3.24]),
    ([[1,10]], [0.]),
])
def test_predict_raw_days(raw, expected):
    check(call('predict_raw_days', np.array(raw), np.array([2.,20.]),
               np.array([1.,10.]), np.array([.36,.72]), 1.08), np.array(expected))


def test_train_scaled_and_validate_example():
    x, y, vx, vy, w = data()
    check(call('train_scaled_and_validate', x,y,vx,vy,w,0.,.1,2),
          (np.array([.36,.72]),1.08,5.7344,18.432))


def test_validation_cannot_change_training():
    x, y, vx, vy, w = data()
    result = call('train_scaled_and_validate', x,y,vx,vy,w,0.,.1,2)
    changed = call('train_scaled_and_validate', x,y,np.array([[4,40]]),np.array([5.]),w,0.,.1,2)
    check(changed[:3], result[:3])
    check(changed[3], 3.0976)
    targets_only = call('train_scaled_and_validate', x,y,vx,np.array([0.,0.]),w,0.,.1,2)
    check(targets_only[:3], result[:3])
    check(targets_only[3], 5.832)


def test_pipeline_zero_steps_uses_training_rulers():
    w = np.array([2.])
    result = call('train_scaled_and_validate', np.array([[1],[3]]),np.array([0.,4.]),
                  np.array([[4],[4],[4]]),np.array([6.,6.,6.]),w,2.,.1,0)
    check(result, (np.array([2.]),2.,0.,0.))
    assert not np.shares_memory(result[0], w)
