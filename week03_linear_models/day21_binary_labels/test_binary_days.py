"""Threshold boundaries, label dtypes, accuracy, and day-level masks."""
import numpy as np
import pytest
import binary_days as s


def check(a,e):
    if isinstance(e,tuple):
        assert isinstance(a,tuple) and len(a)==len(e)
        for x,y in zip(a,e):
            check(x,y)
    elif isinstance(e,np.ndarray):
        assert isinstance(a,np.ndarray) and a.shape==e.shape
        assert np.issubdtype(a.dtype,np.integer)
        np.testing.assert_array_equal(a,e)
    else:
        assert isinstance(a,float)
        assert a==pytest.approx(e,abs=1e-10)


def call(name,*args):
    before=[x.copy() if isinstance(x,np.ndarray) else None for x in args]
    result=getattr(s,name)(*args)
    for x,old in zip(args,before):
        if old is not None:
            np.testing.assert_array_equal(x,old)
    return result


@pytest.mark.parametrize('revenue,cutoff,expected',[
    ([80.,100.,140.,60.],100.,[0,1,1,0]),
    ([0.],0.,[1]),
    ([49.,50.,51.],50.,[0,1,1]),
])
def test_revenue_labels(revenue,cutoff,expected):
    check(call('revenue_labels',np.array(revenue),cutoff),np.array(expected))


@pytest.mark.parametrize('probs,threshold,expected',[
    ([.2,.5,.9,.7],.5,[0,1,1,1]),
    ([.7,.8,.9],.8,[0,1,1]),
    ([0.,1.],1.,[0,1]),
])
def test_probability_labels(probs,threshold,expected):
    check(call('probability_labels',np.array(probs),threshold),np.array(expected))


@pytest.mark.parametrize('pred,actual,expected',[
    ([0,1,1,1],[0,1,1,0],.75),
    ([0,1],[0,1],1.),
    ([1],[0],0.),
])
def test_label_accuracy(pred,actual,expected):
    check(call('label_accuracy',np.array(pred),np.array(actual)),expected)


@pytest.mark.parametrize('ids,pred,actual,expected',[
    ([101,102,103,104],[0,1,1,1],[0,1,1,0],[104]),
    ([9,2,5],[1,0,0],[0,0,1],[9,5]),
    ([42],[1],[1],[]),
])
def test_incorrect_day_ids(ids,pred,actual,expected):
    check(call('incorrect_day_ids',np.array(ids),np.array(pred),np.array(actual)),np.array(expected,dtype=int))


@pytest.mark.parametrize('threshold,expected',[
    (.5,(.75,np.array([104]))),
    (.8,(.75,np.array([102]))),
    (.7,(.5,np.array([102,104]))),
])
def test_classification_report(threshold,expected):
    check(call('classification_report',np.array([101,102,103,104]),np.array([80.,100.,140.,60.]),
               np.array([.2,.5,.9,.7]),100.,threshold),expected)


def test_default_threshold_and_all_correct():
    check(call('probability_labels',np.array([.2,.5])),np.array([0,1]))
    check(call('classification_report',np.array([9,2]),np.array([49.,50.]),np.array([.2,.5]),50.),
          (1.,np.array([],dtype=int)))
