"""Day 23 — How much probability did the model give the real answer? (~35–40 minutes).

Continue the high-revenue-day classifier: label 1 means high revenue, label 0
means not high revenue. p always means estimated probability of label 1.
Today connect probability of the TRUE label to a loss. No gradients or training.
All arrays are finite, nonempty, 1D, and aligned by day; labels are integer 0/1.
For exercises 1,2,4 all p are strictly between 0 and 1. Exercise 3 accepts true
probabilities in (0,1]. Exercise 5 scores are within [-10,10]. No clipping or
extreme-number handling today. Keep inputs unchanged; repeat operations in 1–4.
"""
from typing import Tuple
import numpy as np


def negative_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """1 — Review the meaning of a binary probability (~5 minutes).

    Learn:
    Every supplied p is the model's probability of class 1, the high-revenue
    category. There are only two mutually exclusive categories, so probability
    of class 0 is 1-p. If the model gives class 1 probability 0.8, it gives
    class 0 probability 0.2. These two probabilities sum to 1 for the SAME day.
    Do not sum probabilities across different days or apply a 0.5 threshold.
    Return probabilities, not class labels. A model estimate can still be wrong.

    Independent API reminder:
        values = np.array([2.,5.,8.])
        print(10. - values)  # [8.,5.,2.], scalar broadcasts

    Task: Return each day's model probability of class 0.
    Example input/output:
        probabilities = np.array([0.8,0.3,0.6])
        Expected output: np.array([0.2,0.7,0.4])

    Pause: does p=0.8 mean an 80% chance of class 0 or class 1?
    """
    return 1 - probabilities


def true_label_probabilities(probabilities: np.ndarray, actual_labels: np.ndarray) -> np.ndarray:
    """2 — Select the probability assigned to the real answer (~8 minutes).

    Learn:
    p still describes class 1. After observing the true label, find the model's
    probability for THAT class: use p when actual label is 1, and 1-p when it
    is 0. This is for scoring an existing prediction, not changing that prediction.
    For p=0.8 and actual label 1, the true-class probability is 0.8.
    For the same p=0.8 but actual label 0, it is 0.2: the model favored the wrong
    class. The two days have the same model output but deserve different scores.
    Copy probabilities, make a mask for actual label 0, and replace only those
    positions with their complementary probabilities. Keep the input unchanged.
    Do not threshold probabilities into labels before doing this calculation.

    Independent API reminder:
        values = np.array([3.,-2.,8.])
        result = values.copy()
        result[result < 0.] = 0.
        print(result)  # [3.,0.,8.]
        print(values)  # [3.,-2.,8.]

    Task: Return the probability assigned to the actual class on each day.
    Example input/output:
        probabilities = np.array([0.8,0.3,0.6])
        actual_labels = np.array([1,0,0])
        Expected output: np.array([0.8,0.7,0.4])

    Pause: if actual label is 0, why would using p itself score the wrong class?
    """
    ret = probabilities.copy()
    ret[actual_labels == 1] = probabilities[actual_labels == 1]
    ret[actual_labels == 0] = 1 - probabilities[actual_labels == 0]
    return  ret



def per_day_loss(true_probabilities: np.ndarray) -> np.ndarray:
    """3 — New: negative log turns low probability into a large penalty (~10 minutes).

    Learn:
    Let q be the probability assigned to the TRUE class. The loss for that day
    is -log(q), using the natural logarithm np.log. q is positive and at most 1.
    log is the inverse of exp: log(exp(x))=x. For 0<q<1, log(q) is negative,
    so putting a minus sign in front makes a positive penalty.
    q=1 gives loss 0; q=0.5 gives about 0.693; q=0.1 gives about 2.303.
    Thus giving little probability to what actually happened is penalized more.
    There is no 0.5 cutoff in this loss: q=0.9 scores better than q=0.6 even
    though both are above 0.5. As q approaches zero the penalty grows very large.
    We exclude q=0 today because log(0) is not finite. No clipping is needed.
    Return one loss per day, not an average yet. Smaller loss is better.

    Independent API example — natural logarithm only:
        values = np.array([1.,2.,4.])
        print(np.log(values))  # approximately [0.,0.69314718,1.38629436]

    Task: Return negative natural log of each supplied true-class probability.
    Example input/output:
        true_probabilities = np.array([1.,0.5,0.1])
        Expected output: np.array([0.,0.6931471805599453,2.302585092994046])

    Pause: which deserves more loss, q=0.9 or q=0.1, and why?
    """
    return -np.log(true_probabilities)


def binary_cross_entropy(probabilities: np.ndarray, actual_labels: np.ndarray) -> float:
    """4 — Average the true-class penalties (~8 minutes).

    Learn:
    Repeat exercise 2 to obtain true-class probabilities: p for label 1 and
    1-p for label 0. Apply -log to each, THEN average over days. Return a Python
    float. This is binary cross entropy (BCE), a loss used for binary classifiers.
    It measures how much probability the model assigned to the observed answers.
    There is no need to memorize a longer formula today; follow those steps.
    Do not take the log of the average probability: log is nonlinear, so
    averaging probabilities first would be a different calculation.
    Do not square errors or round probabilities. Unlike accuracy, BCE rewards
    better probability estimates even when thresholded labels do not change.
    Lower BCE is better; higher accuracy is better. They measure different things.

    Independent API reminder:
        values = np.array([2.,4.,9.])
        print(float(np.mean(values)))  # 5.0

    Task: Return the mean negative log probability assigned to the true labels.
    Example input/output:
        probabilities = np.array([0.8,0.25])
        actual_labels = np.array([1,0])
        Expected output: 0.25541281188299536

    Pause: is the true-class probability on the second day 0.25 or 0.75?
    """
    labels = probabilities.copy()
    labels[actual_labels == 1] = labels[actual_labels == 1]
    labels[actual_labels == 0] = 1 - labels[actual_labels == 0]
    return np.mean(-np.log(labels))


def score_classifier(scores: np.ndarray, actual_labels: np.ndarray) -> Tuple[float, float]:
    """5 — Small integration: accuracy and loss from the same model (~8 minutes).

    Learn:
    First convert scores to probabilities with yesterday's sigmoid formula,
    p = 1 / (1 + exp(-scores)). Then evaluate the same probabilities two ways.
    For accuracy, predict 1 when p>=0.5 and compare predicted labels with truth.
    For BCE, keep the original probabilities and score the true class with
    -log(p) or -log(1-p), then average. You may reuse exercise 4 for BCE.
    Return accuracy first and mean BCE second, both Python floats.
    Do not pass thresholded 0/1 predictions to BCE: they discard confidence
    and can produce log(0). Actual labels stay unchanged in both calculations.
    With true label 1, probabilities 0.6 and 0.9 both predict correctly, but
    0.9 gives a lower loss. Accuracy alone cannot distinguish their confidence.
    There is no fitting, parameter update, or threshold search here.

    Independent API reminder:
        flags = np.array([True,False,True])
        print(float(flags.mean()))  # 2/3

    Task: Return accuracy at threshold 0.5 and mean binary cross entropy from supplied scores.
    Example input/output:
        scores = np.array([-1.,0.,1.])
        actual_labels = np.array([0,0,1])
        Expected output: (2/3, 0.43989018519879704)

    Pause: can two models have identical accuracy but different BCE?
    """
    sigmoid = 1 / (1 + np.exp(-scores))
    # accuracy
    pred = sigmoid.copy()
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    accuracy = float(np.mean((pred == actual_labels).astype(int)))

    # BCE
    labels = sigmoid.copy()
    mask = actual_labels == 0
    labels[mask] = 1 - labels[actual_labels == 0]
    return accuracy, float(np.mean(-np.log(labels)))
