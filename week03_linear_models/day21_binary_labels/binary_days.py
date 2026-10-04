"""Day 21 — Is this a high-revenue day? (~35–40 minutes).

A gentle bridge from regression to binary classification. Regression predicts
HOW MUCH revenue; classification predicts which of two categories a day belongs
to. Today 1 means high revenue and 0 means not high revenue.
Model probabilities are supplied; do not train a model or invent probabilities
from standardized features. Sigmoid and classification losses come later.
Use comparisons, masks, and means again. All arrays are nonempty, finite, 1D,
and aligned by day. Probabilities are in [0,1]; thresholds are valid. Leave inputs
unchanged. Repeat direct operations in 1–4; exercise 5 may reuse your functions.
"""
from typing import Tuple
import numpy as np


def revenue_labels(actual_revenue: np.ndarray, revenue_cutoff: float) -> np.ndarray:
    """1 — Turn known revenue into a known category (~7 minutes).

    Learn:
    A label is the answer we want a classifier to predict. Here label 1 means
    actual revenue is at least revenue_cutoff; label 0 means it is below.
    The equality boundary counts as 1. For cutoff 100, revenue 120 has label 1
    and revenue 80 has label 0. These are TRUE labels from observed revenue,
    not model predictions. Do not standardize the revenue before comparing.
    Comparing an array gives a boolean array (N,). Convert True/False to
    integer 1/0 with .astype(int), keeping one entry per day and the same order.

    Independent API reminder:
        flags = np.array([False, True, True])
        print(flags.astype(int))  # [0,1,1]
        print(np.array([2.,5.,8.]) > 5.)  # [False,False,True]

    Task: Return integer labels marking actual revenue at or above the cutoff.
    Example input/output:
        actual_revenue = np.array([80.,100.,140.,60.])
        revenue_cutoff = 100.
        Expected output: np.array([0,1,1,0])

    Pause: does label 1 mean revenue is exactly 1, or membership in a category?
    """
    flags = actual_revenue >= revenue_cutoff
    return flags.astype(int)


def probability_labels(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """2 — Turn the model's probabilities into predicted categories (~7 minutes).

    Learn:
    Each supplied probability is the model's estimate that the day's label is 1.
    A probability of 0.8 is neither revenue 0.8 nor a final class label. Use a
    decision rule: probability at least threshold predicts 1, otherwise 0.
    Equality counts as 1. Return an integer array (N,). The default threshold
    is 0.5; always honor the supplied threshold instead of hardcoding it.
    A high probability is not a guarantee: a model can confidently be wrong.
    The probability threshold has no revenue units; it is different from the
    revenue cutoff used to define the actual categories in exercise 1.

    Independent API reminder:
        values = np.array([3.,7.,9.])
        keep = values < 8.
        print(keep)              # [True,True,False]
        print(keep.astype(int))  # [1,1,0]

    Task: Return integer predicted labels using the supplied probability threshold.
    Example input/output:
        probabilities = np.array([0.2,0.5,0.9,0.7])
        threshold = 0.5
        Expected output: np.array([0,1,1,1])

    Pause: if threshold rises to 0.8, which of these days are still predicted as 1?
    """
    flags = probabilities >= threshold
    return flags.astype(int)


def label_accuracy(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> float:
    """3 — A new score made from a familiar boolean mean (~7 minutes).

    Learn:
    Accuracy is the fraction of days whose predicted category equals the true
    category. Both arrays already contain integer 0/1 labels, shape (N,).
    Compare the arrays element by element, then average the resulting booleans.
    NumPy treats True as 1 and False as 0 when taking a mean. Three correct
    answers out of four gives 3/4=0.75. Return a Python float in [0,1], not 75.
    Higher accuracy is better, unlike MSE where lower is better. Accuracy says
    whether categories matched; it does not measure revenue prediction error
    or the quality of probability estimates. Do not compare probabilities here.

    Independent API reminder:
        flags = np.array([True, False, True])
        print(float(np.mean(flags)))  # 2/3

    Task: Return the fraction of predicted labels equal to the actual labels.
    Example input/output:
        predicted_labels = np.array([0,1,1,1])
        actual_labels = np.array([0,1,1,0])
        Expected output: 0.75

    Pause: if all categories match, should this score be 0 or 1?
    """
    flags = predicted_labels == actual_labels
    return float(np.mean(flags))


def incorrect_day_ids(day_ids: np.ndarray, predicted_labels: np.ndarray,
                      actual_labels: np.ndarray) -> np.ndarray:
    """4 — Repeat masks: which individual days were misclassified? (~6 minutes).

    Learn:
    Compare predicted and actual LABELS per day. Inequality makes a boolean
    mask (N,), then boolean indexing selects matching day IDs in input order.
    Do not use the overall accuracy as a mask: one score cannot identify which
    days were wrong. This repeats the same daily-versus-summary distinction
    you practiced with squared errors, but now each prediction is a category.
    Return integer IDs; no errors means an integer array of shape (0,), not
    a list. IDs are labels for days, not the array positions to return.

    Independent API reminder:
        ids = np.array([9,4,7])
        keep = np.array([True,False,True])
        print(ids[keep])  # [9,7]

    Task: Return IDs of days whose predicted label differs from their actual label.
    Example input/output:
        day_ids = np.array([101,102,103,104])
        predicted_labels = np.array([0,1,1,1])
        actual_labels = np.array([0,1,1,0])
        Expected output: np.array([104])

    Pause: what is the mask's shape compared with the scalar accuracy?
    """
    mask = predicted_labels != actual_labels
    return day_ids[mask]


def classification_report(day_ids: np.ndarray, actual_revenue: np.ndarray,
                          probabilities: np.ndarray, revenue_cutoff: float,
                          probability_threshold: float = 0.5
                          ) -> Tuple[float, np.ndarray]:
    """5 — Small integration: the truth rule and prediction rule are different (~8 minutes).

    Learn:
    Define true labels using actual_revenue and revenue_cutoff. Separately
    define predicted labels using probabilities and probability_threshold.
    Both rules include equality. Compare those two label arrays to report
    accuracy and incorrectly classified day IDs. You may reuse exercises 1–4.
    Keep the two thresholds separate: a revenue cutoff of 100 is in money
    units, while a probability threshold of 0.5 is a decision on [0,1].
    The probabilities are already supplied for these days. This function does
    not fit or tune anything. Thresholds are supplied too; do not search for
    one that makes this dataset's answers look better.
    Changing the probability threshold changes predicted labels but leaves
    true labels fixed. The returned score is a Python float; IDs are an integer
    array in input order, including an empty array when every label is correct.

    Independent API reminder:
        first = np.array([1,0,1])
        second = np.array([1,1,0])
        print(first == second)  # [True,False,False]

    Task: Return classification accuracy and incorrectly classified day IDs.
    Example input/output:
        day_ids = np.array([101,102,103,104])
        actual_revenue = np.array([80.,100.,140.,60.])
        probabilities = np.array([0.2,0.5,0.9,0.7])
        revenue_cutoff = 100.
        probability_threshold = 0.5
        Expected output: (0.75, np.array([104]))

    Pause: why do actual revenue and model probability use different thresholds?
    """
    r_flags = actual_revenue >= revenue_cutoff
    p_flags = probabilities >= probability_threshold
    r_labels = r_flags.astype(int)
    p_labels = p_flags.astype(int)

    return float(np.mean(r_labels == p_labels)), day_ids[r_labels != p_labels]
