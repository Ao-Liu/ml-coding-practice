"""Keep features, predictions, errors, and summary scores distinct."""
import numpy as np
import pytest
import validation_report as s


def check(a, e):
    if isinstance(e, tuple):
        assert isinstance(a, tuple) and len(a) == len(e)
        for x, y in zip(a, e):
            check(x, y)
    elif isinstance(e, np.ndarray):
        assert isinstance(a, np.ndarray) and a.shape == e.shape
        if np.issubdtype(e.dtype, np.integer):
            assert np.issubdtype(a.dtype, np.integer)
        else:
            assert np.issubdtype(a.dtype, np.floating)
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, e, atol=1e-10)
    else:
        assert isinstance(a, float)
        assert a == pytest.approx(e, abs=1e-10)


def call(name, *args):
    before = [x.copy() if isinstance(x,np.ndarray) else None for x in args]
    with np.errstate(divide='raise',invalid='raise'):
        result = getattr(s,name)(*args)
    for x, old in zip(args,before):
        if old is not None:
            np.testing.assert_array_equal(x,old)
    return result


@pytest.mark.parametrize('train,n,expected', [
    ([1.,5.],3,[3.,3.,3.]), ([1,2],1,[1.5]), ([9.],2,[9.,9.]),
])
def test_baseline_predictions(train,n,expected):
    check(call('baseline_predictions',np.array(train),n),np.array(expected))


@pytest.mark.parametrize('sales,means,scales,w,b,expected', [
    ([[2,7],[4,7],[1,7]],[2.,7.],[1.,1.],[2.,0.],3.,[3.,7.,1.]),
    ([[8,12]],[4.,9.],[2.,1.],[2.,3.],1.,[14.]),
])
def test_revenue_predictions(sales,means,scales,w,b,expected):
    check(call('revenue_predictions',np.array(sales),np.array(means),np.array(scales),np.array(w),b),
          np.array(expected))


@pytest.mark.parametrize('actual,base,model,be,me', [
    ([4.,6.,2.],[3.,3.,3.],[3.,7.,1.],[1.,9.,1.],[1.,1.,1.]),
    ([3.],[1.],[5.],[4.],[4.]),
])
def test_daily_squared_errors(actual,base,model,be,me):
    check(call('daily_squared_errors',np.array(actual),np.array(base),np.array(model)),
          (np.array(be),np.array(me)))


@pytest.mark.parametrize('be,me,expected', [
    ([1.,9.,1.],[1.,1.,1.],(11/3,1.,8/3)),
    ([1.,1.],[9.,1.],(1.,5.,-4.)),
    ([4.,0.],[0.,4.],(2.,2.,0.)),
])
def test_comparison_scores(be,me,expected):
    check(call('comparison_scores',np.array(be),np.array(me)),expected)


def report(actual):
    return call('evaluate_days',np.array([101,102,103]),np.array([[2,7],[4,7],[1,7]]),
                np.array(actual),np.array([1.,5.]),np.array([2.,7.]),np.array([1.,1.]),
                np.array([2.,0.]),3.)


def test_evaluate_days_example():
    check(report([4.,6.,2.]),(11/3,1.,8/3,np.array([102])))


def test_evaluate_days_changed_targets_keep_predictors_fixed():
    check(report([3.,3.,3.]),(0.,20/3,-20/3,np.array([],dtype=int)))


def test_evaluate_days_single_row_nonunit_scale():
    check(call('evaluate_days',np.array([42]),np.array([[8,12]]),np.array([12.]),
               np.array([2.,6.]),np.array([4.,9.]),np.array([2.,1.]),np.array([2.,3.]),1.),
          (64.,4.,60.,np.array([42])))


def test_evaluate_days_more_wins_can_still_mean_worse_mse():
    # Baseline guesses 0; fixed model predicts [1,1,10]. Two wins, one large loss.
    check(call('evaluate_days',np.array([8,3,5]),np.array([[1],[1],[10]]),np.array([1.,1.,0.]),
               np.array([-1.,1.]),np.array([0.]),np.array([1.]),np.array([1.]),0.),
          (2/3,100/3,-98/3,np.array([8,3])))
