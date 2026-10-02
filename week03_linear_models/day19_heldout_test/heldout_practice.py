"""Day 19 — Pick on validation, then score the SAME model on test (~35–40 minutes).

New idea: validation selects a model; a separate test set assesses that fixed
choice. No new NumPy API, training loop, or gradient formula today.
Every prediction table has MODELS as rows and DAYS as columns. Validation and
test tables have the same models in the same row order, but may have different
numbers of days. Predictions are supplied from already fitted models.
All arrays are finite; there is at least one model and one day in each set.
Do not modify inputs. Use direct NumPy in 1–4; exercise 5 may reuse helpers.
"""
from typing import Tuple
import numpy as np


def validation_mses(predictions: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """1 — Review: one validation MSE per candidate (~6 minutes).

    Learn:
    predictions is (K,N): each row belongs to one model and contains its
    predictions for N validation days. actual is (N,), the answers for those
    same days. Subtracting actual broadcasts across all K model rows.
    Square each error, then average across DAYS within each row, axis=1.
    The resulting (K,) array keeps model order. Do not average all models
    together; we need a separate score to decide which model to select.
    A model row [2,6] compared with answers [3,4] has errors [-1,2], squared
    errors [1,4], and MSE 2.5. Repeat that meaning for each candidate row.

    Independent API reminder:
        table = np.array([[2.,4.,6.], [1.,3.,8.]])
        print(table.mean(axis=1))  # [4.,4.]

    Task: Return each candidate's validation MSE in a float array.
    Example input/output:
        predictions = np.array([[3.,3.,3.], [3.,7.,1.]])
        actual = np.array([4.,6.,2.])
        Expected output: np.array([11/3, 1.0])

    Pause: which axis counts models, and which counts validation days?
    """
    signed_err = predictions - actual
    return np.mean(signed_err**2, axis=1)


def selected_row(predictions: np.ndarray, model_index: int) -> np.ndarray:
    """2 — Review: an index identifies a MODEL, not a day (~6 minutes).

    Learn:
    A (K,N) table holds K candidate rows. predictions[model_index] retrieves
    all N daily predictions from that one model, giving shape (N,).
    In contrast, predictions[:, model_index] would retrieve a column: several
    models' guesses for one day. That is a different question.
    model_index is supplied and valid; do not find a new winner here.
    Return an independent copy so editing the result cannot change the table.
    Selecting one NumPy row normally creates a view that shares input memory;
    .copy() makes an independent array. Keep shape (N,) even for a single day.

    Independent API reminder:
        values = np.array([2.,5.])
        copied = values.copy()
        copied[0] = 9.
        print(values)  # [2.,5.]
        print(copied)  # [9.,5.]

    Task: Return an independent copy of the supplied model's prediction row.
    Example input/output:
        predictions = np.array([[3.,3.,3.], [3.,7.,1.]])
        model_index = 1
        Expected output: np.array([3.,7.,1.])

    Pause: why does selecting model 1 keep predictions for ALL three days?
    """
    return predictions[model_index].copy()


def choose_on_validation(predictions: np.ndarray, actual: np.ndarray
                         ) -> Tuple[int, np.ndarray]:
    """3 — Connect the validation score to its prediction row (~7 minutes).

    Learn:
    Compute one MSE per candidate, use argmin to get the best model index,
    then retrieve that model's predictions. Return the index AND an independent
    copy of its row. This makes the relationship between idx and selected clear.
    np.argmin returns a position, not the MSE value. Convert it to a Python int.
    If two models tie, choose the first row. Do not choose a different model
    for each day: one model is selected based on its average validation error.
    The returned row is a prediction array, not the array of candidate MSEs.

    Independent API reminder:
        scores = np.array([5.,2.,2.])
        print(int(np.argmin(scores)))  # 1: first minimum's index

    Task: Return the first best model's index and an independent copy of its validation predictions.
    Example input/output:
        predictions = np.array([[3.,3.,3.], [3.,7.,1.]])
        actual = np.array([4.,6.,2.])
        Expected output: (1, np.array([3.,7.,1.]))

    Pause: are you indexing the prediction table or the MSE array to get daily predictions?
    """
    signed_err = predictions - actual
    mses = np.mean(signed_err ** 2, axis=1)
    best_idx = np.argmin(mses)
    return int(best_idx), predictions[best_idx].copy()


def score_on_test(test_predictions: np.ndarray, test_actual: np.ndarray,
                  selected_index: int) -> float:
    """4 — New: score the chosen model without choosing again (~7 minutes).

    Learn:
    Validation already chose selected_index. The test table holds predictions
    from those SAME models in the SAME row order, but for different days.
    Retrieve the row with that index and calculate its MSE against test_actual.
    No argmin belongs here. Do not switch to another row because it happens
    to score better on test: then test answers would help choose the model.
    Validation is used for selection; test is reserved for assessing the fixed
    choice. Repeatedly tuning after seeing test results undermines that role.
    The chosen model can do worse on test. Honest evaluation keeps that result.
    The test table can have a different column count than the validation table;
    the model's row index stays the same because model order is preserved.

    Independent API reminder:
        table = np.array([[2.,5.], [4.,9.], [1.,7.]])
        print(table[2])  # [1.,7.], all predictions from model 2

    Task: Return test MSE for the supplied selected model index as a Python float.
    Example input/output:
        test_predictions = np.array([[5.,5.], [8.,2.]])
        test_actual = np.array([5.,5.])
        selected_index = 1
        Expected output: 9.0

    Pause: row 0 has zero test error here; why must the result still use row 1?
    """
    pred = test_predictions[selected_index].copy()
    return float(np.mean((pred - test_actual) ** 2))


def validate_then_test(validation_predictions: np.ndarray, validation_actual: np.ndarray,
                       test_predictions: np.ndarray, test_actual: np.ndarray,
                       train_actual: np.ndarray) -> Tuple[int, float, float, float]:
    """5 — Keep selection and final scoring separate (~12 minutes).

    Learn:
    Choose the first minimum validation MSE, then keep that index fixed.
    Report its validation MSE and its test MSE. The tables share model order,
    not necessarily day count. You may reuse exercises 1–4 to connect the steps.
    Test values must not change the selected index or selected validation MSE.

    For a familiar reference, the baseline still predicts mean(train_actual).
    Score this baseline against TEST targets this time, because its comparison
    is with the selected model's TEST MSE. Subtract selected test MSE from
    baseline test MSE to get test improvement. Do not mix validation and test
    scores in that difference: they refer to different days.
    Return selected index, selected validation MSE, selected test MSE, and test
    improvement. The index is a Python int; the other three values are Python
    floats. Keep negative improvement values. No fitting occurs in this function.

    Independent API reminder:
        predictions = np.array([2.,6.])
        answers = np.array([3.,4.])
        print(predictions - answers)  # [-1.,2.]

    Task: Return the validation-selected index, its validation MSE, its test MSE, and baseline-minus-model test improvement.
    Example input/output:
        validation_predictions = np.array([[3.,3.,3.], [3.,7.,1.]])
        validation_actual = np.array([4.,6.,2.])
        test_predictions = np.array([[5.,5.], [8.,2.]])
        test_actual = np.array([5.,5.])
        train_actual = np.array([1.,5.])
        Expected output: (1, 1.0, 9.0, -5.0)

    Pause: if test_actual changes, which two returned values must stay unchanged?
    """
    v_signed_err = validation_predictions - validation_actual
    v_mses = np.mean(v_signed_err ** 2, axis=1)
    v_idx = np.argmin(v_mses)
    v_pred = validation_predictions[v_idx].copy()
    v_mse = np.mean((v_pred - validation_actual) ** 2)
    t_pred = test_predictions[v_idx].copy()
    t_mse = np.mean((t_pred - test_actual) ** 2)

    train_mean = np.mean(train_actual)
    train_pred = np.full(test_actual.shape[0], train_mean, dtype=float)
    b_mse = np.mean((train_pred - test_actual) ** 2)

    return int(v_idx), float(v_mse), float(t_mse), float(b_mse - t_mse)





