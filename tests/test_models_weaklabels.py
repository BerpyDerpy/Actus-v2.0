"""Weak HOI labels derived from step annotations plus the protocol.

These labels are the only supervision the HOI classifier gets, so the rules that produce them
matter as much as the model: which frames count as a hold, which hand gets credit, and -- most
of all -- which frames are thrown away rather than guessed at.
"""

from __future__ import annotations

import numpy as np
import pytest

from actus.models.hoi import HOI_CLASSES
from actus.models.weaklabels import (
    IGNORE,
    WeakLabelParams,
    apply_verified,
    hold_steps,
    hold_windows,
    label_counts,
    verification_sample,
    weak_labels,
)
from actus.protocol.schema import load_protocol

NONE_I = HOI_CLASSES.index("none")
TOUCH_I = HOI_CLASSES.index("touching")
HOLD_I = HOI_CLASSES.index("holding")

LABEL_NAMES = ("S1", "S2", "S3", "S4", "S5", "S6", "idle", "other")
PAIR_NAMES = ("frac_inside_bbox", "dist_over_diag")
OBJECTS = ("red", "yellow")


@pytest.fixture(scope="module")
def protocol():
    return load_protocol("protocols/bas_boxes.yaml")


def build(labels, *, inside, over_diag, valid=None, fps=10.0):
    """One take's worth of synthetic inputs. ``inside``/``over_diag`` are (N, H, O)."""
    n = len(labels)
    inside = np.asarray(inside, dtype=np.float32)
    over_diag = np.asarray(over_diag, dtype=np.float32)
    pair = np.stack([inside, over_diag], axis=-1)
    valid = np.ones(inside.shape, dtype=bool) if valid is None else np.asarray(valid, bool)
    return {
        "labels": np.array([LABEL_NAMES.index(x) for x in labels], dtype=np.int8),
        "t": np.arange(n, dtype=np.float32) / fps,
        "pair": pair,
        "pair_valid": valid,
        "label_names": LABEL_NAMES,
        "pair_names": PAIR_NAMES,
        "objects": OBJECTS,
    }


def const(n, value, n_h=2, n_o=2):
    return np.full((n, n_h, n_o), value, dtype=np.float32)


# -- reading the protocol -----------------------------------------------------------------------


def test_hold_steps_come_from_the_protocol(protocol):
    assert hold_steps(protocol) == {"red": ("S2", "S3"), "yellow": ("S4", "S5")}


def test_hold_windows_span_pickup_to_release(protocol):
    labels = ["idle"] * 3 + ["S2"] * 4 + ["S3"] * 3 + ["idle"] * 2
    idx = np.array([LABEL_NAMES.index(x) for x in labels])
    windows = hold_windows(idx, LABEL_NAMES, protocol)
    assert windows["red"] == [(3, 10)]
    assert "yellow" not in windows


def test_an_unreleased_pickup_yields_no_window(protocol):
    """If the place step was skipped the hold has no end, so it is not used as supervision."""
    labels = ["idle"] * 2 + ["S2"] * 4 + ["idle"] * 4
    idx = np.array([LABEL_NAMES.index(x) for x in labels])
    assert hold_windows(idx, LABEL_NAMES, protocol) == {}


def test_repeated_pickups_get_separate_windows(protocol):
    labels = ["S2"] * 2 + ["S3"] * 2 + ["idle"] * 2 + ["S2"] * 2 + ["S3"] * 2
    idx = np.array([LABEL_NAMES.index(x) for x in labels])
    assert hold_windows(idx, LABEL_NAMES, protocol) == {"red": [(0, 4), (6, 10)]}


# -- labelling ----------------------------------------------------------------------------------


def test_the_nearer_hand_gets_the_hold(protocol):
    labels = ["idle"] * 4 + ["S2"] * 6 + ["S3"] * 6 + ["idle"] * 4
    n = len(labels)
    over = const(n, 5.0)
    over[:, 1, 0] = 0.3  # the right hand is on the red box throughout
    inside = np.zeros((n, 2, 2), dtype=np.float32)
    inside[:, 1, 0] = 0.9
    data = build(labels, inside=inside, over_diag=over)
    weak = weak_labels(protocol=protocol, params=WeakLabelParams(boundary_margin_s=0.0), **data)

    mid = 8  # inside the hold window
    assert weak[mid, 1, 0] == HOLD_I
    assert weak[mid, 0, 0] == NONE_I  # the left hand is nowhere near
    assert weak[2, 1, 0] == TOUCH_I  # in contact before the pickup step begins
    assert weak[mid, 1, 1] == NONE_I  # the yellow box is untouched


def test_boundary_frames_are_dropped(protocol):
    labels = ["idle"] * 6 + ["S2"] * 6 + ["S3"] * 6 + ["idle"] * 6
    n = len(labels)
    over = const(n, 5.0)
    over[:, 1, 0] = 0.3
    data = build(labels, inside=np.zeros((n, 2, 2), np.float32), over_diag=over, fps=10.0)
    weak = weak_labels(protocol=protocol, params=WeakLabelParams(boundary_margin_s=0.3), **data)
    # the window opens at index 6 (t = 0.6 s), so 0.3 s either side is ignored
    assert np.all(weak[3:10, 1, 0] == IGNORE)
    assert weak[12, 1, 0] == HOLD_I


def test_unobserved_pairs_are_ignored_not_labelled(protocol):
    labels = ["S2"] * 5 + ["S3"] * 5
    n = len(labels)
    over = const(n, 0.3)
    valid = np.ones((n, 2, 2), dtype=bool)
    valid[:, :, 0] = False  # the red box was never detected
    data = build(labels, inside=const(n, 0.9), over_diag=over, valid=valid)
    weak = weak_labels(protocol=protocol, params=WeakLabelParams(boundary_margin_s=0.0), **data)
    assert np.all(weak[:, :, 0] == IGNORE)
    assert np.all(weak[:, :, 1] != IGNORE)


def test_a_hold_with_no_hand_anywhere_near_is_ignored(protocol):
    """The annotation says it is held; if no hand is close, we decline to guess which."""
    labels = ["S2"] * 5 + ["S3"] * 5
    n = len(labels)
    data = build(labels, inside=np.zeros((n, 2, 2), np.float32), over_diag=const(n, 9.0))
    weak = weak_labels(protocol=protocol, params=WeakLabelParams(boundary_margin_s=0.0), **data)
    assert np.all(weak[:, :, 0] == IGNORE)


def test_contact_without_a_hold_window_is_touching(protocol):
    labels = ["idle"] * 10
    n = len(labels)
    inside = np.zeros((n, 2, 2), dtype=np.float32)
    inside[:, 0, 1] = 0.8  # the left hand rests on the yellow box
    data = build(labels, inside=inside, over_diag=const(n, 5.0))
    weak = weak_labels(protocol=protocol, params=WeakLabelParams(boundary_margin_s=0.0), **data)
    assert np.all(weak[:, 0, 1] == TOUCH_I)
    assert np.all(weak[:, 1, 1] == NONE_I)


def test_label_counts_add_up(protocol):
    labels = ["idle"] * 4 + ["S2"] * 4 + ["S3"] * 4
    n = len(labels)
    over = const(n, 5.0)
    over[:, 1, 0] = 0.3
    data = build(labels, inside=np.zeros((n, 2, 2), np.float32), over_diag=over)
    weak = weak_labels(protocol=protocol, params=WeakLabelParams(boundary_margin_s=0.0), **data)
    counts = label_counts(weak)
    assert sum(counts.values()) == weak.size
    assert counts["holding"] > 0


# -- verification ---------------------------------------------------------------------------


def test_verification_sample_is_class_balanced(protocol):
    labels = ["idle"] * 10 + ["S2"] * 10 + ["S3"] * 10
    n = len(labels)
    over = const(n, 5.0)
    over[:, 1, 0] = 0.3
    inside = np.zeros((n, 2, 2), dtype=np.float32)
    inside[:5, 0, 1] = 0.9
    data = build(labels, inside=inside, over_diag=over)
    weak = weak_labels(protocol=protocol, params=WeakLabelParams(boundary_margin_s=0.0), **data)
    rows = verification_sample(
        weak,
        take="op01_correct_01",
        frame_idx=np.arange(n),
        hands=("left", "right"),
        objects=OBJECTS,
        n=12,
        rng=np.random.default_rng(0),
    )
    drawn = {r["weak_label"] for r in rows}
    assert {"none", "touching", "holding"} <= drawn
    assert all(r["true_label"] == "" for r in rows)


def test_apply_verified_only_marks_filled_rows():
    weak = np.full((5, 2, 2), NONE_I, dtype=np.int8)
    verified = [
        {"take": "t1", "frame_idx": "2", "hand": "right", "object": "red", "true_label": "holding"},
        {"take": "t1", "frame_idx": "3", "hand": "right", "object": "red", "true_label": ""},
        {
            "take": "other",
            "frame_idx": "4",
            "hand": "right",
            "object": "red",
            "true_label": "holding",
        },
    ]
    truth, mask = apply_verified(
        weak,
        take="t1",
        frame_idx=np.arange(5),
        hands=("left", "right"),
        objects=OBJECTS,
        verified=verified,
    )
    assert mask.sum() == 1
    assert truth[2, 1, 0] == HOLD_I
    assert np.all(truth[~mask] == IGNORE)
