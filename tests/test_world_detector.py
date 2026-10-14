"""Automated tests for YOLO-World open-vocabulary detector backend."""

import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.schemas import Detection, ObjectDetector
from src.vision.detector import YOLODetector
from src.vision.world_detector import (
    DEFAULT_YOLO_WORLD_CLASSES,
    YOLOWorldDetector,
)


class TestYOLOWorldPromptValidation:
    """Tests for prompt vocabulary validation and cleaning."""

    def test_prompt_validation_valid_list(self):
        prompts = [" pen ", "keys", "Water Bottle"]
        result = YOLOWorldDetector.validate_prompts(prompts)
        assert result == ["pen", "keys", "Water Bottle"]

    def test_prompt_validation_string_input(self):
        raw_str = "pen, keys; laptop\ncoffee mug,   pen"
        result = YOLOWorldDetector.validate_prompts(raw_str)
        assert result == ["pen", "keys", "laptop", "coffee mug"]

    def test_prompt_validation_deduplication_case_insensitive(self):
        prompts = ["Book", "pen", "BOOK", "Pen", "calculator"]
        result = YOLOWorldDetector.validate_prompts(prompts)
        assert result == ["Book", "pen", "calculator"]

    def test_prompt_validation_empty_list_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            YOLOWorldDetector.validate_prompts([])

    def test_prompt_validation_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            YOLOWorldDetector.validate_prompts(["  ", "\n", ""])

    def test_prompt_validation_invalid_type_raises(self):
        with pytest.raises(ValueError, match="must be a list of strings"):
            YOLOWorldDetector.validate_prompts(12345)  # type: ignore


@pytest.fixture(scope="module")
def world_detector():
    if not os.path.exists("yolov8s-worldv2.pt"):
        pytest.skip("yolov8s-worldv2.pt checkpoint not present in workspace")
    return YOLOWorldDetector(
        model_path="yolov8s-worldv2.pt",
        confidence_threshold=0.25,
        device="cpu",  # Use CPU in unit tests for universal repeatability
    )


class TestYOLOWorldDetectorBackend:
    """Tests for YOLO-World initialization, execution, and protocol compatibility."""

    def test_protocol_compatibility(self, world_detector):
        assert isinstance(world_detector, ObjectDetector)
        assert hasattr(world_detector, "detect")
        assert hasattr(world_detector, "set_classes")
        assert hasattr(world_detector, "get_classes")

    def test_missing_checkpoint_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            YOLOWorldDetector(model_path="nonexistent_checkpoint_xyz.pt")

    def test_initial_classes_loaded(self, world_detector):
        classes = world_detector.get_classes()
        assert len(classes) >= 15
        assert "pen" in classes
        assert "keys" in classes
        assert "wallet" in classes
        assert "charger" in classes
        assert "earbuds" in classes
        assert "calculator" in classes
        assert "glasses" in classes
        assert "notebook" in classes
        assert "mug" in classes
        assert "book" in classes
        assert "bottle" in classes
        assert "phone" in classes
        assert "laptop" in classes

    def test_dynamic_prompt_update(self, world_detector):
        new_prompts = ["scissors", "coffee mug", "reading glasses"]
        updated = world_detector.set_classes(new_prompts)
        assert updated == ["scissors", "coffee mug", "reading glasses"]
        assert world_detector.get_classes() == ["scissors", "coffee mug", "reading glasses"]

        # Restore default classes for subsequent tests
        world_detector.reset_classes()
        assert "pen" in world_detector.get_classes()

    def test_dynamic_prompt_update_with_comma_string(self, world_detector):
        updated = world_detector.set_classes("laptop, wireless mouse, usb cable")
        assert updated == ["laptop", "wireless mouse", "usb cable"]
        world_detector.reset_classes()

    def test_dynamic_prompt_update_empty_rejected(self, world_detector):
        orig_classes = world_detector.get_classes()
        with pytest.raises(ValueError, match="cannot be empty"):
            world_detector.set_classes("   ,   ")
        # Ensure previous classes were preserved
        assert world_detector.get_classes() == orig_classes

    def test_empty_and_invalid_frame_handling(self, world_detector):
        assert world_detector.detect(None) == []
        assert world_detector.detect(np.array([])) == []
        empty_canvas = np.zeros((0, 0, 3), dtype=np.uint8)
        assert world_detector.detect(empty_canvas) == []

    def test_detection_schema_and_sorting(self, world_detector):
        # Run on a solid dummy canvas (expect 0 detections or valid detections)
        canvas = np.zeros((320, 320, 3), dtype=np.uint8)
        results = world_detector.detect(canvas, confidence=0.25)
        assert isinstance(results, list)
        for det in results:
            assert isinstance(det, Detection)
            assert isinstance(det.class_id, int)
            assert isinstance(det.label, str)
            assert 0.0 <= det.confidence <= 1.0
            assert len(det.bbox_xyxy) == 4
            assert det.bbox_xyxy[0] <= det.bbox_xyxy[2]
            assert det.bbox_xyxy[1] <= det.bbox_xyxy[3]

    def test_confidence_filtering(self, world_detector):
        # Mock underlying predict output to test threshold boundary enforcement
        mock_box_low = MagicMock()
        mock_box_low.conf = [MagicMock(item=lambda: 0.20)]
        mock_box_low.cls = [MagicMock(item=lambda: 0)]
        mock_box_low.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([10, 10, 50, 50])))]

        mock_box_high = MagicMock()
        mock_box_high.conf = [MagicMock(item=lambda: 0.85)]
        mock_box_high.cls = [MagicMock(item=lambda: 1)]
        mock_box_high.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([60, 60, 120, 120])))]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box_low, mock_box_high]

        with patch.object(world_detector._model, "predict", return_value=[mock_result]):
            # Threshold 0.50 should exclude the 0.20 box
            dets = world_detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), confidence=0.50)
            assert len(dets) == 1
            assert dets[0].confidence == 0.85

    def test_model_error_graceful_handling(self, world_detector):
        with patch.object(world_detector._model, "predict", side_effect=RuntimeError("CUDA out of memory")):
            results = world_detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
            assert results == []


class TestDetectorFallbackBehavior:
    """Verifies that YOLO-World and standard YOLODetector both conform to ObjectDetector and can fallback."""

    def test_both_implement_object_detector_protocol(self):
        if os.path.exists("yolov8s-worldv2.pt"):
            world_det = YOLOWorldDetector(model_path="yolov8s-worldv2.pt", device="cpu")
            assert isinstance(world_det, ObjectDetector)

        if os.path.exists("yolo11n.pt"):
            standard_det = YOLODetector(model_name="yolo11n.pt", device="cpu")
            assert isinstance(standard_det, ObjectDetector)
            # Both take (frame, confidence)
            dummy = np.zeros((160, 160, 3), dtype=np.uint8)
            res = standard_det.detect(dummy, confidence=0.5)
            assert isinstance(res, list)
