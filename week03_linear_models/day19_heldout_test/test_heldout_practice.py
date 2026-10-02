"""Keep model identity fixed across validation and test, with independent row copies."""
import numpy as np
import pytest
import heldout_practice as s


def check(a,e):
    if isinstance(e,tuple):
        assert isinstance(a,tuple) and len(a)==len(e)
        for x,y in zip(a,e):
            check(x,y)
    elif isinstance(e,np.ndarray):
        assert isinstance(a,np.ndarray) and a.shape==e.shape
        assert np.issubdtype(a.dtype,np.floating)
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a,e,atol=1e-10)
    elif isinstance(e,int):
        assert type(a) is int and a==e
    else:
        assert isinstance(a,float)
        assert a==pytest.approx(e,abs=1e-10)


def call(name,*args):
    copies=[x.copy() if isinstance(x,np.ndarray) else None for x in args]
    result=getattr(s,name)(*args)
    for x,old in zip(args,copies):
        if old is not None:
            np.testing.assert_array_equal(x,old)
    return result


@pytest.mark.parametrize('pred,actual,expected',[
    ([[3.,3.,3.],[3.,7.,1.]],[4.,6.,2.],[11/3,1.]),
    ([[2.],[4.],[7.]],[3.],[1.,1.,16.]),
])
def test_validation_mses(pred,actual,expected):
    check(call('validation_mses',np.array(pred),np.array(actual)),np.array(expected))


@pytest.mark.parametrize('pred,idx,expected',[
    ([[3.,3.,3.],[3.,7.,1.]],1,[3.,7.,1.]),
    ([[2.],[4.]],0,[2.]),
])
def test_selected_row(pred,idx,expected):
    table=np.array(pred)
    row=call('selected_row',table,idx)
    check(row,np.array(expected))
    assert not np.shares_memory(row,table)


@pytest.mark.parametrize('pred,actual,idx,expected',[
    ([[3.,3.,3.],[3.,7.,1.]],[4.,6.,2.],1,[3.,7.,1.]),
    ([[0.,2.],[2.,0.]],[0.,0.],0,[0.,2.]),
    ([[4.]],[3.],0,[4.]),
])
def test_choose_on_validation(pred,actual,idx,expected):
    table=np.array(pred)
    result=call('choose_on_validation',table,np.array(actual))
    check(result,(idx,np.array(expected)))
    assert not np.shares_memory(result[1],table)


@pytest.mark.parametrize('pred,actual,idx,expected',[
    ([[5.,5.],[8.,2.]],[5.,5.],1,9.),
    ([[9.],[4.]],[3.],0,36.),
])
def test_score_on_test(pred,actual,idx,expected):
    check(call('score_on_test',np.array(pred),np.array(actual),idx),expected)


def report(test_pred,test_actual,train_actual=(1.,5.)):
    return call('validate_then_test',np.array([[3.,3.,3.],[3.,7.,1.]]),np.array([4.,6.,2.]),
                np.array(test_pred),np.array(test_actual),np.array(train_actual))


def test_validate_then_test_example():
    check(report([[5.,5.],[8.,2.]],[5.,5.]),(1,1.,9.,-5.))


def test_test_targets_do_not_reselect_model():
    check(report([[5.,5.],[8.,2.]],[8.,2.]),(1,1.,0.,13.))


def test_test_predictions_do_not_reselect_model():
    check(report([[5.],[10.]],[5.]),(1,1.,25.,-21.))


def test_train_targets_only_change_baseline():
    check(report([[5.,5.],[8.,2.]],[5.,5.],(5.,5.)),(1,1.,9.,-9.))


def test_validation_tie_first_model_even_when_test_prefers_second():
    check(call('validate_then_test',np.array([[0.,2.],[2.,0.]]),np.array([0.,0.]),
               np.array([[9.],[3.]]),np.array([3.]),np.array([1.,5.])),(0,2.,36.,-36.))


def test_one_candidate_can_beat_baseline():
    check(call('validate_then_test',np.array([[2.,4.]]),np.array([2.,4.]),
               np.array([[3.]]),np.array([3.]),np.array([0.,0.])),(0,0.,0.,9.))


def test_baseline_uses_test_day_count_not_training_day_count():
    check(report([[5.,5.],[8.,2.]],[5.,5.],(1.,3.,5.)),(1,1.,9.,-5.))
