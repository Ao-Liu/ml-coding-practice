"""Candidate/day axes, first-index ties, baseline direction, and unchanged inputs."""
import numpy as np
import pytest
import model_selection as s


def check(a,e):
    if isinstance(e,tuple):
        assert isinstance(a,tuple) and len(a)==len(e)
        for x,y in zip(a,e):
            check(x,y)
    elif isinstance(e,np.ndarray):
        assert isinstance(a,np.ndarray) and a.shape==e.shape
        kind = np.integer if np.issubdtype(e.dtype,np.integer) else np.floating
        assert np.issubdtype(a.dtype,kind)
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a,e,atol=1e-10)
    elif isinstance(e,int):
        assert type(a) is int and a==e
    else:
        assert isinstance(a,float)
        assert a==pytest.approx(e,abs=1e-10)


def call(name,*args):
    copies=[x.copy() if isinstance(x,np.ndarray) else None for x in args]
    with np.errstate(divide='raise',invalid='raise'):
        result=getattr(s,name)(*args)
    for x,old in zip(args,copies):
        if old is not None:
            np.testing.assert_array_equal(x,old)
    return result


@pytest.mark.parametrize('raw,means,scales,w,b,expected',[
    ([[2,7],[4,7],[1,7]],[2.,7.],[1.,1.],[2.,0.],3.,[3.,7.,1.]),
    ([[8,12]],[4.,9.],[2.,1.],[2.,3.],1.,[14.]),
])
def test_predict_candidate(raw,means,scales,w,b,expected):
    check(call('predict_candidate',np.array(raw),np.array(means),np.array(scales),np.array(w),b),np.array(expected))


@pytest.mark.parametrize('pred,actual,expected',[
    ([[3.,3.,3.],[3.,7.,1.]],[4.,6.,2.],[11/3,1.]),
    ([[2.],[4.],[7.]],[3.],[1.,1.,16.]),
    ([[1.,5.,4.]],[2.,3.,4.],[5/3]),
])
def test_candidate_mses(pred,actual,expected):
    check(call('candidate_mses',np.array(pred),np.array(actual)),np.array(expected))


@pytest.mark.parametrize('mses,expected',[
    ([11/3,1.,2.],1),([4.,2.,2.],1),([.5],0),([4.,3.,.5],2),
])
def test_best_candidate(mses,expected):
    check(call('best_candidate',np.array(mses)),expected)


@pytest.mark.parametrize('actual,pred,train,expected',[
    ([4.,6.,2.],[3.,7.,1.],[1.,5.],[102]),
    ([3.,3.,3.],[3.,3.,3.],[1.,5.],[]),
])
def test_improved_ids(actual,pred,train,expected):
    check(call('improved_ids',np.array([101,102,103]),np.array(actual),np.array(pred),np.array(train)),
          np.array(expected,dtype=int))


def test_selection_report_example():
    check(call('selection_report',np.array([101,102,103]),
               np.array([[3.,3.,3.],[3.,7.,1.],[4.,8.,0.]]),np.array([4.,6.,2.]),np.array([1.,5.])),
          (1,1.,8/3,np.array([102])))


def test_selection_report_tie_keeps_first_prediction_row():
    check(call('selection_report',np.array([9,4]),np.array([[0.,2.],[2.,0.]]),np.array([0.,0.]),np.array([-1.,3.])),
          (0,2.,-1.,np.array([9])))


def test_selection_report_single_day_all_candidates_worse():
    check(call('selection_report',np.array([7]),np.array([[9.],[5.]]),np.array([3.]),np.array([1.,5.])),
          (1,4.,-4.,np.array([],dtype=int)))


def test_selection_report_training_targets_change_only_baseline():
    args=(np.array([101,102,103]),np.array([[3.,3.,3.],[3.,7.,1.]]),np.array([4.,6.,2.]))
    check(call('selection_report',*args,np.array([0.,0.])),
          (1,1.,53/3,np.array([101,102,103])))
