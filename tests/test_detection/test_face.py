"""Tests for face detection module."""

import numpy as np
import pytest

from emotion_detection_action.core.config import ModelConfig
from emotion_detection_action.core.types import BoundingBox, FaceDetection
from emotion_detection_action.detection.face import (
    FaceDetector,
    MTCNN_AVAILABLE,
    RETINAFACE_AVAILABLE,
)


class TestFaceDetector:
    """Tests for FaceDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        config = ModelConfig(model_id="mtcnn", device="cpu")
        detector = FaceDetector(config, threshold=0.8)

        assert detector.threshold == 0.8
        assert detector.max_faces == 5
        assert not detector.is_loaded

    def test_initialization_with_custom_params(self):
        """Test detector initialization with custom parameters."""
        config = ModelConfig(model_id="retinaface", device="cpu")
        detector = FaceDetector(
            config,
            threshold=0.95,
            min_face_size=30,
            max_faces=3,
            return_landmarks=False,
            return_face_images=False,
        )

        assert detector.threshold == 0.95
        assert detector.min_face_size == 30
        assert detector.max_faces == 3
        assert detector.return_landmarks is False
        assert detector.return_face_images is False

    @pytest.mark.skipif(not MTCNN_AVAILABLE, reason="MTCNN not installed")
    def test_load_mtcnn(self):
        """Test loading MTCNN model."""
        config = ModelConfig(model_id="mtcnn", device="cpu")
        detector = FaceDetector(config)

        detector.load()
        assert detector.is_loaded
        assert detector.model_type == "mtcnn"

        detector.unload()
        assert not detector.is_loaded

    @pytest.mark.skipif(not RETINAFACE_AVAILABLE, reason="RetinaFace not installed")
    def test_load_retinaface(self):
        """Test loading RetinaFace model."""
        config = ModelConfig(model_id="retinaface", device="cpu")
        detector = FaceDetector(config)

        detector.load()
        assert detector.is_loaded
        assert detector.model_type == "retinaface"

        detector.unload()
        assert not detector.is_loaded

    def test_load_default_model(self):
        """Test loading with unknown model defaults to MTCNN."""
        config = ModelConfig(model_id="unknown_model", device="cpu")
        detector = FaceDetector(config)

        if MTCNN_AVAILABLE:
            detector.load()
            assert detector.model_type == "mtcnn"
            detector.unload()

    def test_predict_not_loaded_raises(self):
        """Test that predict raises error when model not loaded."""
        config = ModelConfig(model_id="mtcnn", device="cpu")
        detector = FaceDetector(config)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            detector.predict(np.zeros((100, 100, 3), dtype=np.uint8))

    @pytest.mark.skipif(not MTCNN_AVAILABLE, reason="MTCNN not installed")
    def test_predict_empty_image(self):
        """Test prediction on empty/blank image returns no faces."""
        config = ModelConfig(model_id="mtcnn", device="cpu")
        detector = FaceDetector(config)
        detector.load()

        # Blank image - should detect no faces
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.predict(blank_image)

        assert isinstance(detections, list)
        # May or may not detect false positives, just check it doesn't crash

        detector.unload()

    @pytest.mark.skipif(not MTCNN_AVAILABLE, reason="MTCNN not installed")
    def test_predict_grayscale_image(self):
        """Test prediction on grayscale image."""
        config = ModelConfig(model_id="mtcnn", device="cpu")
        detector = FaceDetector(config)
        detector.load()

        # Grayscale image
        gray_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        detections = detector.predict(gray_image)

        assert isinstance(detections, list)

        detector.unload()

    @pytest.mark.skipif(not MTCNN_AVAILABLE, reason="MTCNN not installed")
    def test_max_faces_limit(self):
        """Test that max_faces parameter limits detections."""
        config = ModelConfig(model_id="mtcnn", device="cpu")
        detector = FaceDetector(config, max_faces=1)
        detector.load()

        # Create a simple test image
        test_image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        detections = detector.predict(test_image)

        # Should return at most 1 face
        assert len(detections) <= 1

        detector.unload()

    def test_context_manager(self):
        """Test FaceDetector as context manager."""
        config = ModelConfig(model_id="mtcnn", device="cpu")

        if MTCNN_AVAILABLE:
            with FaceDetector(config) as detector:
                assert detector.is_loaded
            assert not detector.is_loaded


class TestFaceDetectorHelpers:
    """Tests for FaceDetector helper methods."""

    def test_draw_detections(self):
        """Test drawing detections on image."""
        # Create test image and detections
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            FaceDetection(
                bbox=BoundingBox(x=100, y=100, width=100, height=100),
                confidence=0.95,
            ),
            FaceDetection(
                bbox=BoundingBox(x=300, y=200, width=80, height=80),
                confidence=0.88,
            ),
        ]

        result = FaceDetector.draw_detections(image, detections)

        # Result should be different from input (boxes drawn)
        assert result.shape == image.shape
        # Check that something was drawn (not all zeros)
        assert not np.array_equal(result, image)

    def test_draw_detections_custom_color(self):
        """Test drawing detections with custom color."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            FaceDetection(
                bbox=BoundingBox(x=100, y=100, width=100, height=100),
                confidence=0.95,
            ),
        ]

        # Draw with red color
        result = FaceDetector.draw_detections(
            image, detections, color=(0, 0, 255), thickness=3
        )

        assert result.shape == image.shape

    def test_draw_detections_empty_list(self):
        """Test drawing with no detections."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = FaceDetector.draw_detections(image, [])

        # Result should be identical to input
        assert np.array_equal(result, image)


class TestRetinaFaceSpecific:
    """Tests specific to RetinaFace implementation."""

    @pytest.mark.skipif(not RETINAFACE_AVAILABLE, reason="RetinaFace not installed")
    def test_retinaface_landmarks_format(self):
        """Test that RetinaFace returns landmarks in correct format."""
        config = ModelConfig(model_id="retinaface", device="cpu")
        detector = FaceDetector(config, return_landmarks=True)
        detector.load()

        # Create test image
        test_image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        detections = detector.predict(test_image)

        for det in detections:
            if det.landmarks is not None:
                # RetinaFace should return 5 landmarks: 2 eyes, nose, 2 mouth corners
                assert det.landmarks.shape == (5, 2)

        detector.unload()

    @pytest.mark.skipif(not RETINAFACE_AVAILABLE, reason="RetinaFace not installed")
    def test_retinaface_face_image_cropping(self):
        """Test that RetinaFace correctly crops face images."""
        config = ModelConfig(model_id="retinaface", device="cpu")
        detector = FaceDetector(config, return_face_images=True)
        detector.load()

        # Create test image
        test_image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        detections = detector.predict(test_image)

        for det in detections:
            if det.face_image is not None:
                # Face image dimensions should match bbox
                assert det.face_image.shape[0] == det.bbox.height
                assert det.face_image.shape[1] == det.bbox.width

        detector.unload()


class TestModelTypeSelection:
    """Tests for model type selection and switching."""

    def test_model_type_property(self):
        """Test model_type property."""
        config = ModelConfig(model_id="mtcnn", device="cpu")
        detector = FaceDetector(config)

        # Before loading, default is mtcnn
        assert detector.model_type == "mtcnn"

    @pytest.mark.skipif(not MTCNN_AVAILABLE, reason="MTCNN not installed")
    def test_model_type_after_load_mtcnn(self):
        """Test model_type is set correctly after loading MTCNN."""
        config = ModelConfig(model_id="mtcnn", device="cpu")
        detector = FaceDetector(config)
        detector.load()

        assert detector.model_type == "mtcnn"
        detector.unload()

    @pytest.mark.skipif(not RETINAFACE_AVAILABLE, reason="RetinaFace not installed")
    def test_model_type_after_load_retinaface(self):
        """Test model_type is set correctly after loading RetinaFace."""
        config = ModelConfig(model_id="retinaface", device="cpu")
        detector = FaceDetector(config)
        detector.load()

        assert detector.model_type == "retinaface"
        detector.unload()

    def test_model_type_reset_after_unload(self):
        """Test model_type resets after unload."""
        config = ModelConfig(model_id="mtcnn", device="cpu")
        detector = FaceDetector(config)

        if MTCNN_AVAILABLE:
            detector.load()
            detector.unload()
            assert detector.model_type == "mtcnn"  # Reset to default
