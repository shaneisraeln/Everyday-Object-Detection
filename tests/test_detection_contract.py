"""Tests for Detection schema, overlay rendering, and detector contract."""

import numpy as np
import pytest
from src.schemas import Detection
from src.vision.overlay import render_detections_overlay


def test_detection_dataclass():
    det = Detection(
        class_id=39,
        label="bottle",
        confidence=0.8842,
        bbox_xyxy=(10, 20, 100, 200),
    )
    assert det.label == "bottle"
    assert det.confidence == 0.8842
    assert det.bbox_xyxy == (10, 20, 100, 200)

    d_dict = det.to_dict()
    assert d_dict["label"] == "bottle"
    assert d_dict["confidence"] == 0.8842


def test_render_detections_overlay_empty():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = render_detections_overlay(img, [])
    assert res.shape == (100, 100, 3)
    assert np.array_equal(res, img)


def test_render_detections_overlay_with_boxes():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    det1 = Detection(class_id=0, label="bottle", confidence=0.92, bbox_xyxy=(10, 10, 50, 50))
    det2 = Detection(class_id=1, label="cup", confidence=0.75, bbox_xyxy=(60, 60, 120, 120))

    annotated = render_detections_overlay(img, [det1, det2], active_index=0)
    assert annotated.shape == (200, 200, 3)
    # The image should have non-zero pixels where the overlay was drawn
    assert np.count_nonzero(annotated) > 0


def test_detection_sorting():
    d1 = Detection(0, "cup", 0.55, (0, 0, 10, 10))
    d2 = Detection(1, "bottle", 0.91, (0, 0, 10, 10))
    d3 = Detection(2, "book", 0.78, (0, 0, 10, 10))
    dets = [d1, d2, d3]
    dets.sort(key=lambda d: d.confidence, reverse=True)
    assert [d.label for d in dets] == ["bottle", "book", "cup"]
