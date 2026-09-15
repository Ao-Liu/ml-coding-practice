"""Tests for the linked Day 2 exercises; no reference implementations."""

import numpy as np
import pytest
import gradebook as g


@pytest.fixture
def data():
    return (np.array([101, 102, 103, 104]),
            np.array([[50, 70, 90], [80, 60, 70], [90, 40, 80], [60, 80, 70]]))


def array_equal(actual, expected):
    assert isinstance(actual, np.ndarray)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected)


def test_course_means(data):
    _, scores = data
    array_equal(g.course_means(scores), [70., 62.5, 77.5])


def test_student_means(data):
    _, scores = data
    array_equal(g.student_means(scores), [70., 70., 70., 70.])
    array_equal(g.student_means(np.array([[10, 20, 30], [40, 60, 80]])), [20., 60.])


def test_top_student(data):
    ids, scores = data
    assert g.top_student(ids, scores) == 101
    assert g.top_student(np.array([90, 12, 77]), np.array([[1, 2], [9, 10], [3, 4]])) == 12


def test_passing_mask(data):
    _, scores = data
    mask = g.passing_mask(scores)
    array_equal(mask, [False, True, False, True])
    assert mask.dtype == np.bool_
    array_equal(g.passing_mask(scores, pass_mark=70), [False, False, False, False])


def test_select_students(data):
    ids, scores = data
    selected_ids, selected_scores = g.select_students(ids, scores, np.array([False, True, False, True]))
    array_equal(selected_ids, [102, 104])
    array_equal(selected_scores, [[80, 60, 70], [60, 80, 70]])


def test_select_students_empty(data):
    ids, scores = data
    selected_ids, selected_scores = g.select_students(ids, scores, np.zeros(4, dtype=bool))
    array_equal(selected_ids, np.empty(0, dtype=int))
    array_equal(selected_scores, np.empty((0, 3)))


def test_adjust_scores(data):
    _, scores = data
    array_equal(g.adjust_scores(scores, np.array([10, 0, 5])),
                [[60, 70, 95], [90, 60, 75], [100, 40, 85], [70, 80, 75]])


def test_adjust_scores_clip_and_fractions():
    array_equal(g.adjust_scores(np.array([[98, 2, 40], [50, 10, 70]]), np.array([5, -8, .5])),
                [[100, 0, 40.5], [55, 2, 70.5]])


def test_center_courses(data):
    _, scores = data
    array_equal(g.center_courses(scores), [[-20, 7.5, 12.5], [10, -2.5, -7.5],
                                          [20, -22.5, 2.5], [-10, 17.5, -7.5]])
    array_equal(g.center_courses(np.array([[10, 30]])), [[0., 0.]])


def check_report(report, ids, means, courses):
    assert set(report) == {'student_ids', 'student_means', 'course_means'}
    array_equal(report['student_ids'], ids)
    array_equal(report['student_means'], means)
    if courses is None:
        assert report['course_means'] is None
    else:
        array_equal(report['course_means'], courses)


def test_build_report(data):
    ids, scores = data
    check_report(g.build_report(ids, scores, np.array([10, 0, 5])),
                 [101, 102, 104], [75., 75., 75.], [220/3, 70., 245/3])


def test_build_report_no_passers(data):
    ids, scores = data
    with np.errstate(divide='raise', invalid='raise'):
        result = g.build_report(ids, scores, np.zeros(3), pass_mark=100)
    check_report(result, [], [], None)


def test_build_report_single_passer_and_clipping():
    check_report(g.build_report(np.array([8, 9]), np.array([[95, 55], [30, 90]]), np.array([10, 5])),
                 [8], [80.], [100., 60.])


def test_needs_support_optional(data):
    ids, scores = data
    array_equal(g.needs_support(ids, scores), [103])
    array_equal(g.needs_support(ids, scores, threshold=40), [])


@pytest.mark.parametrize('name', [
    'course_means', 'student_means', 'top_student', 'passing_mask',
    'select_students', 'adjust_scores', 'center_courses', 'build_report',
    'needs_support',
])
def test_inputs_unchanged(name, data):
    ids, scores = data
    bonus = np.array([10., 0., 5.])
    mask = np.array([True, False, True, False])
    arguments = {
        'course_means': (scores,), 'student_means': (scores,),
        'top_student': (ids, scores), 'passing_mask': (scores,),
        'select_students': (ids, scores, mask), 'adjust_scores': (scores, bonus),
        'center_courses': (scores,), 'build_report': (ids, scores, bonus),
        'needs_support': (ids, scores),
    }[name]
    snapshots = [a.copy() for a in arguments]
    getattr(g, name)(*arguments)
    for actual, original in zip(arguments, snapshots):
        np.testing.assert_array_equal(actual, original)
