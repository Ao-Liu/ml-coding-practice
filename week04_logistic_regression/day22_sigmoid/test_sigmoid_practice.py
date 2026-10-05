"""Scores, sigmoid values, decision boundaries, and fixed preprocessing."""
import numpy as np
import pytest
import sigmoid_practice as s


def check(a,e):
    if isinstance(e,tuple):
        assert isinstance(a,tuple) and len(a)==len(e)
        for x,y in zip(a,e):
            check(x,y)
    elif isinstance(e,np.ndarray):
        assert isinstance(a,np.ndarray) and a.shape==e.shape
        kind=np.integer if np.issubdtype(e.dtype,np.integer) else np.floating
        assert np.issubdtype(a.dtype,kind)
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a,e,rtol=1e-7,atol=1e-12)
    else:
        assert isinstance(a,float)
        assert a==pytest.approx(e,abs=1e-10)


def call(name,*args):
    before=[x.copy() if isinstance(x,np.ndarray) else None for x in args]
    with np.errstate(over='raise',divide='raise',invalid='raise'):
        result=getattr(s,name)(*args)
    for x,old in zip(args,before):
        if old is not None:
            np.testing.assert_array_equal(x,old)
    return result


@pytest.mark.parametrize('x,w,b,expected',[
    ([[-1.,0.],[0.,1.],[1.,2.]],[1.,0.],0.,[-1.,0.,1.]),
    ([[2.,1.],[0.,3.],[1.,2.]],[2.,-1.],.5,[3.5,-2.5,.5]),
])
def test_linear_scores(x,w,b,expected):
    check(call('linear_scores',np.array(x),np.array(w),b),np.array(expected))


@pytest.mark.parametrize('scores,expected',[
    ([-1.,0.,1.],[.2689414213699951,.5,.7310585786300049]),
    ([-2.,2.],[.11920292202211755,.8807970779778823]),
    ([0.],[.5]),
])
def test_sigmoid_values(scores,expected):
    check(call('sigmoid_values',np.array(scores)),np.array(expected))


@pytest.mark.parametrize('x,w,b,expected',[
    ([[-1.,0.],[0.,1.],[1.,2.]],[1.,0.],0.,[.2689414213699951,.5,.7310585786300049]),
    ([[1.,2.],[3.,4.],[0.,0.]],[1.,-1.],1.,[.5,.5,.7310585786300049]),
])
def test_probability_predictions(x,w,b,expected):
    check(call('probability_predictions',np.array(x),np.array(w),b),np.array(expected))


@pytest.mark.parametrize('scores,threshold,expected',[
    ([-1.,0.,1.],.5,[0,1,1]),
    ([0.,1.,2.],.8,[0,0,1]),
    ([-2.,0.,2.],0.,[1,1,1]),
    ([-2.,0.,2.],1.,[0,0,0]),
])
def test_labels_from_scores(scores,threshold,expected):
    check(call('labels_from_scores',np.array(scores),threshold),np.array(expected))


@pytest.mark.parametrize('threshold,expected',[
    (.5,(2/3,np.array([102]))),
    (.8,(2/3,np.array([103]))),
])
def test_evaluate_sigmoid_model(threshold,expected):
    check(call('evaluate_sigmoid_model',np.array([101,102,103]),np.array([[1,7],[2,7],[3,7]]),
               np.array([2.,7.]),np.array([1.,1.]),np.array([1.,0.]),0.,np.array([0,0,1]),threshold),expected)


def test_single_day_fixed_nonunit_statistics_and_bias():
    check(call('evaluate_sigmoid_model',np.array([42]),np.array([[8,12]]),np.array([4.,9.]),
               np.array([2.,1.]),np.array([1.,-1.]),1.,np.array([1])),(1.,np.array([],dtype=int)))


def test_default_score_threshold():
    check(call('labels_from_scores',np.array([-.1,0.,.1])),np.array([0,1,1]))
