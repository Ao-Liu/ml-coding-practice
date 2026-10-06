"""Class meaning, unchanged probabilities, log-before-mean, and accuracy vs loss."""
import numpy as np
import pytest
import probability_loss as s


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
    else:
        assert isinstance(a,float)
        assert a==pytest.approx(e,abs=1e-10)


def call(name,*args):
    before=[x.copy() for x in args]
    with np.errstate(divide='raise',invalid='raise',over='raise'):
        result=getattr(s,name)(*args)
    for x,old in zip(args,before):
        np.testing.assert_array_equal(x,old)
    return result


@pytest.mark.parametrize('p,expected',[
    ([.8,.3,.6],[.2,.7,.4]),([.5],[.5]),([.01,.99],[.99,.01]),
])
def test_negative_probabilities(p,expected):
    check(call('negative_probabilities',np.array(p)),np.array(expected))


@pytest.mark.parametrize('p,y,expected',[
    ([.8,.3,.6],[1,0,0],[.8,.7,.4]),
    ([.2,.9],[1,1],[.2,.9]),
    ([.2,.9],[0,0],[.8,.1]),
])
def test_true_label_probabilities(p,y,expected):
    check(call('true_label_probabilities',np.array(p),np.array(y)),np.array(expected))


@pytest.mark.parametrize('q,expected',[
    ([1.,.5,.1],[0.,.6931471805599453,2.302585092994046]),
    ([.9],[.10536051565782628]),
    ([.01,.99],[4.605170185988091,.01005033585350145]),
])
def test_per_day_loss(q,expected):
    check(call('per_day_loss',np.array(q)),np.array(expected))


@pytest.mark.parametrize('p,y,expected',[
    ([.8,.25],[1,0],.25541281188299536),
    ([.1,.9],[1,1],1.203972804325936),
    ([.8],[0],1.6094379124341003),
])
def test_binary_cross_entropy(p,y,expected):
    check(call('binary_cross_entropy',np.array(p),np.array(y)),expected)


@pytest.mark.parametrize('scores,y,expected',[
    ([-1.,0.,1.],[0,0,1],(2/3,.43989018519879704)),
    ([0.],[1],(1.,.6931471805599453)),
    ([1.],[1],(1.,.31326168751822286)),
    ([1.],[0],(0.,1.3132616875182228)),
])
def test_score_classifier(scores,y,expected):
    check(call('score_classifier',np.array(scores),np.array(y)),expected)
