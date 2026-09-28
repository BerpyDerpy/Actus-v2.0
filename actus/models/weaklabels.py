"""Weak labels for the HOI classifier, derived from the step annotations plus geometry.

Nobody labelled "this hand is holding the red box" on 20,000 frames. What we do have is the
human step annotation, and the protocol says what a step does to which object. Between the start
of "take out the red box" and the end of "place the red box in Zone A", the red box is in
somebody's hand. That is the whole idea, and it is the one the build plan's T-HOI-TEMP card
specifies.

Deriving the windows from the protocol, not from hard-coded step ids
--------------------------------------------------------------------
A step with ``requires_hoi: X`` is where X is picked up. The hold ends at the end of the first
later step whose ``post`` refers to X again (for ``bas_boxes`` that is "place the red box in
Zone A"). So the windows come out of the protocol YAML, and a second protocol gets its weak
labels for free.

Where geometry comes in
-----------------------
The step annotation says the object is held; it does not say by which hand. That is settled by
the features: within a hold window the hand that is actually near the object is ``holding`` and
the other is ``none`` unless it is also in contact. Outside a hold window, a hand in contact
with the object is ``touching``.

What is deliberately thrown away
--------------------------------
* Pairs where the hand or the object was not observed -- there is nothing to learn from.
* A margin either side of every window boundary. The exact frame a grasp begins is not
  recoverable from a step annotation pressed by a human watching playback, and training on
  those frames teaches the model our reaction time.
* Hold windows in takes where the annotation is inconsistent with the protocol (a step missing
  its partner), rather than guessing at the intent.

The result is noisy supervision, and the model card says so. ``verification_sample`` draws the
frames a human should check by hand so the reported numbers can be anchored to something real.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from actus.models.hoi import HOI_CLASSES, HOLDING, NONE, TOUCHING

IGNORE = -1  # excluded from training and from scoring


@dataclass(frozen=True)
class WeakLabelParams:
    boundary_margin_s: float = 0.4  # dropped either side of a hold window's edges
    touch_inside: float = 0.25  # fraction of hand keypoints inside the grown bbox => contact
    touch_dist_over_diag: float = 1.0  # ...or within one box diagonal of its centre
    hold_near_dist_over_diag: float = 2.0  # the holding hand must at least be near the object


def hold_steps(protocol) -> dict[str, tuple[str, str | None]]:
    """``{object: (pickup step id, release step id or None)}`` read from the protocol."""
    out: dict[str, tuple[str, str | None]] = {}
    order = list(protocol.step_ids)
    for step in protocol.steps:
        obj = step.requires_hoi
        if obj is None:
            continue
        release = None
        for later in order[order.index(step.id) + 1 :]:
            spec = protocol.step(later)
            if any(key.split(".", 1)[0] == obj for key in spec.post):
                release = later
                break
        out[obj] = (step.id, release)
    return out


def _runs(labels: np.ndarray, value: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` index ranges where ``labels == value``."""
    hit = labels == value
    if not hit.any():
        return []
    edges = np.diff(hit.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if hit[0]:
        starts.insert(0, 0)
    if hit[-1]:
        ends.append(len(labels))
    return list(zip(starts, ends, strict=True))


def hold_windows(
    labels: np.ndarray, label_names: Sequence[str], protocol
) -> dict[str, list[tuple[int, int]]]:
    """``{object: [(start, end), ...]}`` index ranges over which the object is held.

    A window runs from the start of the pickup step to the end of the release step. Repeated
    occurrences (an operator who puts the box back and redoes the step) each get their own
    window, paired in order.
    """
    idx = {name: i for i, name in enumerate(label_names)}
    out: dict[str, list[tuple[int, int]]] = {}
    for obj, (pickup, release) in hold_steps(protocol).items():
        if pickup not in idx:
            continue
        picks = _runs(labels, idx[pickup])
        rels = _runs(labels, idx[release]) if release in idx else []
        windows = []
        for ps, _pe in picks:
            after = [r for r in rels if r[0] >= ps]
            if after:
                windows.append((ps, after[0][1]))
            elif not rels:
                # No release step is annotated in this take (e.g. the place step was skipped).
                # The hold cannot be bounded, so it is not used as supervision.
                continue
        if windows:
            out[obj] = windows
    return out


def weak_labels(
    *,
    labels: np.ndarray,
    t: np.ndarray,
    pair: np.ndarray,
    pair_valid: np.ndarray,
    label_names: Sequence[str],
    pair_names: Sequence[str],
    objects: Sequence[str],
    protocol,
    params: WeakLabelParams | None = None,
) -> np.ndarray:
    """``(N, H, O)`` weak class indices into ``HOI_CLASSES``, or ``IGNORE``."""
    params = params or WeakLabelParams()
    n, n_h, n_o = pair_valid.shape
    out = np.full((n, n_h, n_o), IGNORE, dtype=np.int8)
    col = {name: i for i, name in enumerate(pair_names)}
    inside = pair[:, :, :, col["frac_inside_bbox"]]
    over_diag = pair[:, :, :, col["dist_over_diag"]]
    contact = (inside >= params.touch_inside) | (
        (over_diag > 0) & (over_diag <= params.touch_dist_over_diag)
    )
    near = (over_diag > 0) & (over_diag <= params.hold_near_dist_over_diag)

    windows = hold_windows(np.asarray(labels), label_names, protocol)
    held_mask = np.zeros((n, n_o), dtype=bool)
    drop_mask = np.zeros((n, n_o), dtype=bool)
    for o, obj in enumerate(objects):
        for start, end in windows.get(obj, []):
            held_mask[start:end, o] = True
            drop_mask[_margin(t, start, params.boundary_margin_s), o] = True
            drop_mask[_margin(t, end, params.boundary_margin_s), o] = True

    for o in range(n_o):
        for h in range(n_h):
            valid = pair_valid[:, h, o]
            lab = np.full(n, IGNORE, dtype=np.int8)
            lab[valid & ~held_mask[:, o] & contact[:, h, o]] = HOI_CLASSES.index(TOUCHING)
            lab[valid & ~held_mask[:, o] & ~contact[:, h, o]] = HOI_CLASSES.index(NONE)
            in_hold = valid & held_mask[:, o]
            # Inside a hold window the annotation says *someone* holds it; geometry picks who.
            best = _closer_hand(over_diag[:, :, o], near[:, :, o], pair_valid[:, :, o])
            lab[in_hold & (best == h)] = HOI_CLASSES.index(HOLDING)
            lab[in_hold & (best != h) & contact[:, h, o]] = HOI_CLASSES.index(TOUCHING)
            lab[in_hold & (best != h) & ~contact[:, h, o]] = HOI_CLASSES.index(NONE)
            lab[in_hold & (best < 0)] = IGNORE  # held, but no hand is anywhere near it
            lab[drop_mask[:, o]] = IGNORE
            out[:, h, o] = lab
    return out


def _margin(t: np.ndarray, index: int, seconds: float) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    if len(t) == 0:
        return np.zeros(0, dtype=bool)
    centre = t[min(max(index, 0), len(t) - 1)]
    return np.abs(t - centre) <= seconds + 1e-6  # tolerate float32 timestamps


def _closer_hand(over_diag: np.ndarray, near: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Index of the nearer *valid, near enough* hand per frame, or -1."""
    d = np.where(valid & near, over_diag, np.inf)
    best = np.argmin(d, axis=1)
    ok = np.isfinite(np.min(d, axis=1))
    return np.where(ok, best, -1)


def label_counts(weak: np.ndarray) -> dict[str, int]:
    out = {name: int(np.sum(weak == i)) for i, name in enumerate(HOI_CLASSES)}
    out["ignored"] = int(np.sum(weak == IGNORE))
    return out


def verification_sample(
    weak: np.ndarray,
    *,
    take: str,
    frame_idx: np.ndarray,
    hands: Sequence[str],
    objects: Sequence[str],
    n: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Draw pairs for a human to check, balanced across the weak classes.

    Balanced on purpose: ``none`` outnumbers ``holding`` by orders of magnitude, so a uniform
    sample would verify almost nothing about the class that matters.
    """
    rows: list[dict] = []
    per_class = max(1, n // len(HOI_CLASSES))
    for cls_idx, cls in enumerate(HOI_CLASSES):
        hits = np.argwhere(weak == cls_idx)
        if not len(hits):
            continue
        pick = rng.choice(len(hits), size=min(per_class, len(hits)), replace=False)
        for i in pick:
            f, h, o = hits[i]
            rows.append(
                {
                    "take": take,
                    "frame_idx": int(frame_idx[f]),
                    "hand": hands[h],
                    "object": objects[o],
                    "weak_label": cls,
                    "true_label": "",
                }
            )
    return rows


def apply_verified(
    weak: np.ndarray,
    *,
    take: str,
    frame_idx: np.ndarray,
    hands: Sequence[str],
    objects: Sequence[str],
    verified: Sequence[Mapping[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """``(true labels, mask)`` for the verified rows of this take; everything else is IGNORE."""
    pos = {int(v): i for i, v in enumerate(frame_idx)}
    hand_idx = {h: i for i, h in enumerate(hands)}
    obj_idx = {o: i for i, o in enumerate(objects)}
    truth = np.full_like(weak, IGNORE)
    mask = np.zeros(weak.shape, dtype=bool)
    for row in verified:
        if row.get("take") != take or not row.get("true_label"):
            continue
        f = pos.get(int(row["frame_idx"]))
        h = hand_idx.get(row["hand"])
        o = obj_idx.get(row["object"])
        if f is None or h is None or o is None or row["true_label"] not in HOI_CLASSES:
            continue
        truth[f, h, o] = HOI_CLASSES.index(row["true_label"])
        mask[f, h, o] = True
    return truth, mask
