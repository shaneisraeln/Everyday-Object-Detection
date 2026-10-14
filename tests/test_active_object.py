"""Tests for ActiveObjectManager selection logic and freezing during Q&A."""

import pytest
from src.schemas import Detection, ObjectContext
from src.context.active_object import ActiveObjectManager


def test_empty_detections():
    mgr = ActiveObjectManager()
    ctx, msg = mgr.update_from_detections([])
    assert ctx is None
    assert "No objects detected" in msg


def test_single_detection_auto_select():
    mgr = ActiveObjectManager()
    det = Detection(class_id=39, label="bottle", confidence=0.88, bbox_xyxy=(10, 10, 100, 200))
    ctx, msg = mgr.update_from_detections([det])
    assert ctx is not None
    assert ctx.label == "bottle"
    assert ctx.confidence == 0.88
    assert "Active object: bottle" in msg


def test_multiple_detections_require_selection():
    mgr = ActiveObjectManager()
    det1 = Detection(class_id=39, label="bottle", confidence=0.85, bbox_xyxy=(0, 0, 50, 50))
    det2 = Detection(class_id=41, label="cup", confidence=0.82, bbox_xyxy=(60, 60, 100, 100))

    ctx, msg = mgr.update_from_detections([det1, det2])
    # When multiple detections exist without active context, ask user to select
    assert ctx is None
    assert "Please select one" in msg

    # Explicit user selection
    ctx2, msg2 = mgr.update_from_detections([det1, det2], user_selected_index=1)
    assert ctx2 is not None
    assert ctx2.label == "cup"


def test_context_freezing_during_qa():
    mgr = ActiveObjectManager()
    det = Detection(class_id=39, label="bottle", confidence=0.90, bbox_xyxy=(0, 0, 50, 50))
    mgr.update_from_detections([det])

    # Freeze context for Q&A
    frozen_ctx = mgr.freeze_for_qa()
    assert frozen_ctx is not None
    assert frozen_ctx.label == "bottle"
    assert mgr.is_frozen is True

    # New frame comes with a different object while frozen
    other_det = Detection(class_id=63, label="laptop", confidence=0.95, bbox_xyxy=(0, 0, 200, 200))
    ctx_after_frame, msg = mgr.update_from_detections([other_det])

    # Context must NOT switch during Q&A
    assert ctx_after_frame.label == "bottle"
    assert "frozen" in msg.lower()

    # Unfreeze
    mgr.unfreeze_qa()
    assert mgr.is_frozen is False


def test_disappeared_object_retention():
    mgr = ActiveObjectManager()
    det = Detection(class_id=39, label="bottle", confidence=0.85, bbox_xyxy=(0, 0, 50, 50))
    mgr.update_from_detections([det])

    # In next frame, no objects detected
    ctx, msg = mgr.update_from_detections([])
    assert ctx is not None
    assert ctx.label == "bottle"
    assert "no longer visible" in msg
