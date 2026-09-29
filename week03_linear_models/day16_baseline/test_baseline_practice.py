"""Training-only baseline, fair comparisons, masks, and safe preprocessing."""
import numpy as np
import pytest
import baseline_practice as s


def check(a, e):
    if isinstance(e, tuple):
        assert isinstance(a, tuple) and len(a) == len(e)
        for x, y in zip(a, e):
            check(x, y)
    elif isinstance(e, np.ndarray):
        assert isinstance(a, np.ndarray) and a.shape == e.shape and a.dtype == e.dtype
        np.testing.assert_array_equal(a, e)
    else:
        assert isinstance(a, float)
        assert a == pytest.approx(e, abs=1e-9)


def call(name, *args):
    copies = [x.copy() if isinstance(x,np.ndarray) else None for x in args]
    with np.errstate(divide='raise', invalid='raise'):
        result = getattr(s,name)(*args)
    for x, old in zip(args,copies):
        if old is not None:
            np.testing.assert_array_equal(x,old)
    return result


@pytest.mark.parametrize('values,expected', [([1.,3.,8.],4.),([9.],9.)])
def test_baseline_revenue(values,expected):
    check(call('baseline_revenue',np.array(values)),expected)


@pytest.mark.parametrize('train,val,expected', [
    ([1.,3.,8.],[3.,7.],5.), ([2.,6.],[10.],36.), ([4.],[4.,4.],0.),
])
def test_baseline_mse(train,val,expected):
    check(call('baseline_mse',np.array(train),np.array(val)),expected)


@pytest.mark.parametrize('raw,actual,expected', [
    ([[2,7],[4,7]],[3.,7.],.5), ([[4,9]],[8.],4.),
])
def test_model_mse(raw,actual,expected):
    check(call('model_mse',np.array(raw),np.array(actual),np.array([2.,7.]),
               np.array([1.,1.]),np.array([1.5,0.]),3.),expected)


def test_model_mse_nonunit_scale():
    check(call('model_mse',np.array([[8,12]]),np.array([12.]),np.array([4.,9.]),
               np.array([2.,1.]),np.array([2.,3.]),1.),4.)


@pytest.mark.parametrize('actual,pred,base,expected', [
    ([3.,7.,4.],[3.,6.,5.],4.,[101,102]),
    ([3.,7.,4.],[5.,5.,5.],5.,[]),
    ([3.,7.,4.],[3.,7.,4.],0.,[101,102,103]),
])
def test_improved_days(actual,pred,base,expected):
    check(call('improved_days',np.array([101,102,103]),np.array(actual),np.array(pred),base),
          np.array(expected,dtype=int))


@pytest.mark.parametrize('steps,rate,expected', [
    (2,.1,(8.,11.8784,-3.8784)),
    (0,.1,(8.,29.,-21.)),
    (1,.5,(8.,0.,8.)),
])
def test_train_and_compare(steps,rate,expected):
    check(call('train_and_compare',np.array([[1,7],[3,7]]),np.array([1.,5.]),
               np.array([[2,7],[4,7]]),np.array([3.,7.]),np.zeros(2),0.,rate,steps),expected)


def test_training_statistics_and_target_isolation():
    # Validation is one changed row; it must not choose the baseline or scales.
    check(call('train_and_compare',np.array([[2,9],[6,9]]),np.array([1.,5.]),
               np.array([[8,12]]),np.array([10.]),np.zeros(2),0.,.1,2),
          (49.,55.9504,-6.9504))


def test_all_constant_training_bias_still_updates():
    check(call('train_and_compare',np.array([[7],[7],[7]]),np.array([1.,3.,5.]),
               np.array([[7]]),np.array([3.]),np.zeros(1),0.,.1,2),
          (0.,3.6864,-3.6864))
