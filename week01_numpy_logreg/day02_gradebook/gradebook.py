"""Day 2 — One dataset, repeated NumPy practice (~40 minutes + optional work).

Open this file and work from top to bottom. All instructions are here.
Implement ONE function, run its tests, then continue. Leave tests unchanged.

Today's table: rows are students, columns are courses. A separate 1D array
holds student IDs in the SAME row order. The examples share this dataset:

    student_ids = np.array([101, 102, 103, 104])             # (4,)
    scores = np.array([[50, 70, 90],                        # (4, 3)
                       [80, 60, 70],
                       [90, 40, 80],
                       [60, 80, 70]])

Use NumPy, without Python loops or comprehensions. Do not modify inputs.
Assume valid shapes, unique integer IDs, and finite scores. The original
scores table is nonempty. Scores may be integer or floating-point arrays.

Plan:
  1–3 (~10 min): recall mean/axis, then connect positions to student IDs.
  4–5 (~10 min): build one mask and apply it to two related arrays.
  6–7 (~10 min): adjust and center columns using broadcasting.
  8   (~10 min): reuse earlier functions to build a report.
  9   (optional): repeat masking with 'any'.
Take longer if needed; finishing a smaller part yourself is useful practice.

From the repository root:
  python -m pytest -q week01_numpy_logreg/day02_gradebook -k course_means
Replace course_means with your current function name. Unfinished functions
are expected to fail with NotImplementedError.
"""

import numpy as np


def course_means(scores):
    """1 — Recall: one mean per course.

    LEARN: For a 2D table, axis=0 collapses rows and axis=1 collapses
    columns. Before coding, try remembering yesterday's column_means.

    Example: np.mean(np.array([2, 4, 9])) gives 5.0.
    TASK: Given scores (N, D), return course means with shape (D,).
    For today's table -> [70., 62.5, 77.5].
    Check: why should there be THREE results rather than four?

    axis = 0 => 每个col取平均
    axis = 1 => 每个row取平均
    """
    return np.mean(scores, axis=0)


def student_means(scores):
    """2 — Repeat with the other axis: one mean per student.

    LEARN: Same operation, different axis. The dimension you reduce
    disappears. Reducing the D course values leaves N student results.

    TASK: Return shape (N,). For today's table -> [70., 70., 70., 70.].
    Work from memory first. Test cases also include unequal averages.
    Connection: the next function will use YOUR student_means function.
    """
    return np.mean(scores, axis=1)


def top_student(student_ids, scores):
    """3 — New: argmax gives a position, not a value or an ID.

    LEARN:
        values = np.array([4, 9, 2])
        np.max(values)     # 9: largest VALUE
        np.argmax(values)  # 1: POSITION of that value
    You can use a position to index another array in matching order.
    In a tie, np.argmax chooses the first occurrence.

    TASK: Reuse student_means, find the highest average, and return that
    student's ID (a Python or NumPy integer). Do not return the position.
    For today's table -> 101, because all four students tie.
    """
    means = student_means(scores)
    highest_idx = np.argmax(means)
    return student_ids[highest_idx]

def passing_mask(scores, pass_mark=60):
    """4 — New: reduce a boolean table with np.all.

    LEARN: Comparing a whole table gives one boolean per cell.
        flags = np.array([[True, True], [True, False]])
        np.all(flags, axis=1)  # [True, False]
    'all' is True only when every entry along that axis is True.

    TASK: Return a boolean array (N,): True when ALL of that student's
    scores are >= pass_mark. A score equal to pass_mark counts as passing.
    For today's table at 60 -> [False, True, False, True].
    First think of the shape of the comparison, then the reduced mask.
    """
    # 这里就会形成类似[[True, True], [True, False]]
    flags = scores >= pass_mark
    return np.all(flags, axis=1)


def select_students(student_ids, scores, mask):
    """5 — Reuse ONE mask to keep related data aligned.

    LEARN:
        names = np.array(['A', 'B', 'C'])
        keep = np.array([True, False, True])
        names[keep]  # ['A', 'C']
    A 1D boolean mask also selects rows of a 2D table.

    TASK: Return a tuple (selected_ids, selected_scores), applying mask
    to both inputs. Preserve their order. mask has shape (N,).
    With today's passing mask -> IDs [102, 104], scores [[80,60,70],
    [60,80,70]]. Shapes are (2,) and (2,3).
    If nothing matches, shapes must be (0,) and (0,D), not a Python list.
    """
    return student_ids[mask], scores[mask]


def adjust_scores(scores, course_bonus):
    """6 — Revisit broadcasting, then learn np.clip.

    LEARN: (N,D) + (D,) applies one bonus per course across every row.
    np.clip(array, minimum, maximum) limits values to a closed interval:
        np.clip(np.array([-3, 4, 12]), 0, 10)  # [0, 4, 10]

    TASK: Add course_bonus (D,) to scores (N,D), THEN clip to [0,100].
    Return shape (N,D), keeping both inputs unchanged. Bonuses may be
    negative or fractional. Do not round or convert to integer dtype.
    Today's table with bonus [10,0,5] ->
        [[60,70,95], [90,60,75], [100,40,85], [70,80,75]].
    """
    added = scores + course_bonus
    return np.clip(added, 0, 100)


def center_courses(scores):
    """7 — New: keepdims=True makes reduced dimensions explicit.

    LEARN: For a (N,D) table:
        np.mean(scores, axis=0)                 # shape (D,)
        np.mean(scores, axis=0, keepdims=True)  # shape (1,D)
    keepdims retains the reduced axis with length 1. Both shapes can
    broadcast against (N,D); the explicit 1 makes the alignment visible.
    Centering subtracts a feature's mean so it has mean zero.

    TASK: Use mean with keepdims=True to subtract each course's mean
    from that column. Return shape (N,D), without changing scores.
    For [[10,30],[20,50]] -> [[-5.,-10.],[5.,10.]].
    Check: each output column should have mean approximately zero.
    This operation is a step toward ML feature preprocessing.
    """
    means = np.mean(scores, axis=0, keepdims=True)
    return scores - means


def build_report(student_ids, scores, course_bonus, pass_mark=60):
    """8 — Integrated task: build on YOUR earlier functions.

    No new NumPy API here. Reuse adjust_scores, passing_mask,
    select_students, student_means, and course_means.

    TASK, in order:
      1. Apply bonuses and clip scores.
      2. Decide who passes ALL courses using the ADJUSTED scores.
      3. Select those students' IDs and adjusted score rows together.
      4. Calculate personal and course means for this selected group.

    Return a dictionary with exactly these keys:
      'student_ids': selected IDs, shape (M,)
      'student_means': selected students' adjusted averages, shape (M,)
      'course_means': selected group's adjusted course means, shape (D,)

    If M=0, return empty arrays for the first two keys and None for
    'course_means'. Check this case BEFORE computing any means; averaging
    an empty group is undefined. np.array([]) creates an empty 1D array.
    A regular Python if statement is fine here.

    Today's table with bonus [10,0,5] at pass_mark=60 ->
      student_ids:   [101, 102, 104]
      student_means: [75., 75., 75.]
      course_means:  [220/3, 70., 245/3]

    Before coding: sketch the shape after every step. center_courses
    is a separate preprocessing exercise; this report uses actual scores.
    """
    adjusted = adjust_scores(scores, course_bonus)
    masked = passing_mask(adjusted, pass_mark)
    selected_ids, corresponding_scores = select_students(student_ids, adjusted, masked)
    if selected_ids.size == 0:
        return {
            "student_ids": selected_ids,
            "student_means": np.array([]),
            "course_means": None,
        }
    cm = course_means(corresponding_scores)
    sm = student_means(corresponding_scores)
    return {
        "student_ids": selected_ids,
        "student_means": sm,
        "course_means": cm,
    }



def needs_support(student_ids, scores, threshold=50):
    """9 — OPTIONAL repetition: any instead of all (~5 minutes).

    LEARN: np.any is True if at least one entry is True.
        flags = np.array([[False, True], [False, False]])
        np.any(flags, axis=1)  # [True, False]

    TASK: Return IDs of students with ANY course score strictly below
    threshold, in original order. Use the provided scores directly.
    For today's table at 50 -> [103]. A score of exactly 50 is not below.
    Reuse select_students if helpful. Return an empty 1D array if none.
    """
    masked = scores < threshold
    scores_below_threshold = np.any(masked, axis=1) # [True], [False], ...
    selected_ids, _ = select_students(student_ids, scores, scores_below_threshold)
    if selected_ids.size == 0:
        return np.array([])
    return selected_ids
