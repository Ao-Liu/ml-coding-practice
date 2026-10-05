"""Day 22 — From a linear score to a probability (~35–40 minutes).

Yesterday probabilities were supplied. Today compute them with sigmoid, then
reuse thresholding and accuracy. One new function; no training or derivatives.
All arrays are finite and datasets nonempty. Scores stay in [-30,30] today so
the direct exponential formula is safe; extreme-score handling comes later.
Features are prepared unless a function explicitly accepts raw sales. Leave
inputs unchanged. Repeat operations in 1–4; exercise 5 may reuse your functions.
"""
from typing import Tuple
import numpy as np


def linear_scores(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """1 — Review the linear calculation with a new interpretation (~6 minutes).

    Learn:
    Compute features @ weights + bias: (N,D) @ (D,) gives (N,).
    For this classifier, the result is a SCORE, not revenue or a probability.
    Scores can be negative or greater than 1. Higher scores favor label 1.
    Features are already prepared; do not calculate new means or scales.
    These are supplied classification weights, not the earlier revenue model's
    weights. For features [2,1], weights [1,-1], bias 0, the score is 1;
    that weighted combination does not mean the event has probability 1.

    Independent API reminder:
        grid = np.array([[2.,0.], [1.,3.]])
        print(grid @ np.array([4.,2.]))  # [8.,10.]

    Task: Return one linear score per feature row as a float array.
    Example input/output:
        features = np.array([[-1.,0.], [0.,1.], [1.,2.]])
        weights = np.array([1.,0.])
        bias = 0.
        Expected output: np.array([-1.,0.,1.])

    Pause: why is -1 allowed as a score but not as a probability?
    """
    return features @ weights + bias


def sigmoid_values(scores: np.ndarray) -> np.ndarray:
    """2 — New: smoothly convert scores to probabilities (~10 minutes).

    Learn:
    Sigmoid is sigma(z) = 1 / (1 + exp(-z)), separately for every score.
    exp(x) means e raised to x; np.exp computes this for an entire array.
    It is always positive, so the formula maps finite scores into (0,1)
    mathematically. Preserve the input's shape (N,) and return floats.
    At z=0, exp(0)=1, so sigmoid gives 1/(1+1)=0.5.
    At z=1, exp(-1) is about 0.368, giving about 0.731.
    At z=-1, exp(1) is about 2.718, giving about 0.269.
    Bigger scores give bigger probabilities. Negative scores map below 0.5;
    positive scores map above 0.5. Clipping a score into [0,1] is not sigmoid:
    it would throw away distinctions between many different scores.
    The result is a model probability, not a guarantee the class is correct.
    Scores are limited to [-30,30] today; no overflow handling is needed yet.

    sigmoid 是用来把任意实数分数转换到 0 和 1 之间的函数

    Independent API example — exponential only:
        values = np.array([0.,1.,2.])
        print(np.exp(values))  # approximately [1.,2.71828183,7.38905610]

    Task: Return sigmoid probabilities for the supplied scores as a float array.
    Example input/output:
        scores = np.array([-1.,0.,1.])
        Expected output: np.array([0.2689414213699951,0.5,0.7310585786300049])

    Pause: why does a zero score produce 0.5 rather than zero?
    """
    return 1 / (1 + np.exp(-scores))


def probability_predictions(features: np.ndarray, weights: np.ndarray,
                            bias: float) -> np.ndarray:
    """3 — Repeat: features -> scores -> probabilities (~7 minutes).

    Learn:
    First calculate linear scores from the supplied prepared features. Then
    apply sigmoid to those scores. Do not apply sigmoid separately to each
    feature before the weighted sum: it belongs AFTER the sum and bias.
    Shapes are (N,D) -> (N,) -> (N,). Return probabilities, not 0/1 labels yet.
    Actual revenue and true labels are not needed to make these predictions.
    Repeat the operations directly here to practice the new formula again.

    Independent API reminder:
        values = np.array([2.,4.])
        print(1. / values)  # [0.5,0.25], elementwise reciprocal

    Task: Return probabilities from the supplied linear model followed by sigmoid.
    Example input/output:
        features = np.array([[-1.,0.], [0.,1.], [1.,2.]])
        weights = np.array([1.,0.])
        bias = 0.
        Expected output: np.array([0.2689414213699951,0.5,0.7310585786300049])

    Pause: would returning features @ weights + bias alone satisfy this task?
    """
    pred = features @ weights + bias
    return 1 / (1 + np.exp(-pred))


def labels_from_scores(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """4 — Repeat sigmoid and yesterday's threshold rule (~6 minutes).

    Learn:
    Convert scores to probabilities, then predict 1 when probability is at
    least threshold, otherwise 0. Equality counts as 1. Return integer (N,).
    threshold is in [0,1] and is a PROBABILITY cutoff, not a raw score cutoff.
    A score of 0 maps to probability 0.5, so at threshold 0.5 it predicts 1.
    Comparing that raw score directly with 0.5 would incorrectly predict 0.
    At threshold 0.8, a score of 1 (probability about 0.731) predicts 0.
    Honor the supplied threshold instead of hardcoding a score cutoff.

    Independent API reminder:
        flags = np.array([False,True,False])
        print(flags.astype(int))  # [0,1,0]

    Task: Return integer labels after applying sigmoid and the probability threshold.
    Example input/output:
        scores = np.array([-1.,0.,1.])
        threshold = 0.5
        Expected output: np.array([0,1,1])

    Pause: does a positive score guarantee label 1 at every probability threshold?
    """
    sigmoid = 1 / (1 + np.exp(-scores))
    return (sigmoid >= threshold).astype(int)


def evaluate_sigmoid_model(day_ids: np.ndarray, sales: np.ndarray, means: np.ndarray,
                           safe_scales: np.ndarray, weights: np.ndarray, bias: float,
                           actual_labels: np.ndarray, threshold: float = 0.5
                           ) -> Tuple[float, np.ndarray]:
    """5 — Small integration: raw sales to a classification report (~10 minutes).

    Learn:
    Transform raw sales with the supplied TRAINING means and positive safe
    scales. Calculate scores, sigmoid probabilities, then predicted labels.
    Compare labels with actual_labels to get accuracy and incorrect day IDs.
    You may reuse earlier functions; no training loop is needed.
    Keep stages distinct: standardized features are not scores, scores are
    not probabilities, and probabilities are not final 0/1 decisions.
    Zero training std has already been replaced by 1 in the supplied scales.
    Do not re-estimate statistics from these rows or standardize actual_labels:
    true categories are already supplied as integer 0/1 values.
    Accuracy is a Python float in [0,1]. Incorrect IDs are an integer array in
    input order, shape (0,) when all labels match. Build a per-day mismatch
    mask; overall accuracy is a scalar and cannot identify the wrong days.

    Independent API reminder:
        ids = np.array([8,3,5])
        keep = np.array([False,True,False])
        print(ids[keep])  # [3]

    Task: Return classification accuracy and incorrect day IDs for the supplied sigmoid model.
    Example input/output:
        day_ids = np.array([101,102,103])
        sales = np.array([[1,7], [2,7], [3,7]])
        means = np.array([2.,7.])
        safe_scales = np.array([1.,1.])
        weights = np.array([1.,0.])
        bias = 0.
        actual_labels = np.array([0,0,1])
        threshold = 0.5
        Expected output: (2/3, np.array([102]))

    Pause: if actual_labels changes, should the model's probabilities change?
    """
    weights = weights.copy()
    stand = (sales - means) / safe_scales
    pred = stand @ weights + bias
    sigmoid = 1 / (1 + np.exp(-pred))
    pass_threshold = sigmoid >= threshold
    flags = actual_labels == pass_threshold.astype(int)
    return np.mean(flags), day_ids[actual_labels != pass_threshold.astype(int)]
