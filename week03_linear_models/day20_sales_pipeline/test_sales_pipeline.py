"""Chronological alignment, train-only preprocessing/fitting, and unequal set sizes."""
import numpy as np
import pytest
import sales_pipeline as s


def check(a,e):
    if isinstance(e,tuple):
        assert isinstance(a,tuple) and len(a)==len(e)
        for x,y in zip(a,e):
            check(x,y)
    elif isinstance(e,np.ndarray):
        assert isinstance(a,np.ndarray) and a.shape==e.shape
        if np.issubdtype(e.dtype,np.floating):
            assert np.issubdtype(a.dtype,np.floating)
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a,e,atol=1e-10)
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


def test_split_days_example():
    check(call('split_days',np.array([[1,7],[3,7],[2,7],[4,7]]),np.array([1.,5.,3.,7.]),2,1),
          (np.array([[1,7],[3,7]]),np.array([1.,5.]),np.array([[2,7]]),np.array([3.]),
           np.array([[4,7]]),np.array([7.])))


def test_split_days_validation_count_is_not_stop_index():
    check(call('split_days',np.array([[9],[8],[7],[6],[5]]),np.array([90.,80.,70.,60.,50.]),1,3),
          (np.array([[9]]),np.array([90.]),np.array([[8],[7],[6]]),np.array([80.,70.,60.]),
           np.array([[5]]),np.array([50.])))


def test_prepare_three_sets_example():
    check(call('prepare_three_sets',np.array([[1,7],[3,7]]),np.array([[2,7]]),np.array([[4,7],[1,7],[2,9]])),
          (np.array([[-1.,0.],[1.,0.]]),np.array([[0.,0.]]),np.array([[2.,0.],[-1.,0.],[0.,2.]])))


def test_prepare_three_sets_training_ruler_stays_fixed():
    check(call('prepare_three_sets',np.array([[2,9],[6,9]]),np.array([[100,12]]),np.array([[8,10],[4,9]])),
          (np.array([[-1.,0.],[1.,0.]]),np.array([[48.,3.]]),np.array([[2.,1.],[0.,0.]])))


def test_fit_training_rows_two_steps():
    check(call('fit_training_rows',np.array([[-1.,0.],[1.,0.]]),np.array([1.,5.]),np.zeros(2),0.,.1,2),
          (np.array([.72,0.]),1.08))


def test_fit_training_rows_zero_steps_copy():
    w=np.array([2.,4.])
    result=call('fit_training_rows',np.array([[-1.,0.],[1.,0.]]),np.array([1.,5.]),w,3.,.1,0)
    check(result,(np.array([2.,4.]),3.))
    assert not np.shares_memory(result[0],w)


def test_fit_training_rows_all_constant():
    check(call('fit_training_rows',np.zeros((3,1)),np.array([1.,3.,5.]),np.array([4.]),0.,.1,2),
          (np.array([4.]),1.08))


@pytest.mark.parametrize('pred,actual,train,expected',[
    ([7.,1.,3.],[7.,1.,3.],[1.,5.],(20/3,0.,20/3)),
    ([8.,2.],[5.,5.],[1.,3.,5.],(4.,9.,-5.)),
    ([3.],[3.],[1.,5.],(0.,0.,0.)),
])
def test_test_comparison(pred,actual,train,expected):
    check(call('test_comparison',np.array(pred),np.array(actual),np.array(train)),expected)


def pipeline(actual,steps=1):
    return call('run_pipeline',np.array([[1,7],[3,7],[2,7],[4,7],[1,7],[2,9]]),np.array(actual),
                2,1,np.zeros(2),0.,.5,steps)


def test_run_pipeline_example():
    check(pipeline([1.,5.,3.,7.,1.,3.]),(0.,0.,20/3,20/3))


def test_run_pipeline_zero_steps():
    check(pipeline([1.,5.,3.,7.,1.,3.],0),(9.,59/3,20/3,-13.))


def test_test_targets_cannot_change_validation_score():
    check(pipeline([1.,5.,3.,8.,2.,4.]),(0.,1.,9.,8.))


def test_validation_targets_cannot_change_test_scores():
    check(pipeline([1.,5.,30.,7.,1.,3.]),(729.,0.,20/3,20/3))


def test_pipeline_single_constant_training_day():
    check(call('run_pipeline',np.array([[7],[8],[9],[10]]),np.array([2.,3.,4.,5.]),
               1,1,np.array([0.]),0.,.5,1),(1.,6.5,6.5,0.))
