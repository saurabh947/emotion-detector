"""Face detection module using MTCNN or RetinaFace."""

from typing import Any

import numpy as np

from emotion_detection_action.core.config import ModelConfig
from emotion_detection_action.core.types import BoundingBox, FaceDetection
from emotion_detection_action.models.base import BaseModel

# Try to import face detection libraries
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False


class FaceDetector(BaseModel[np.ndarray, list[FaceDetection]]):
    """Face detection using MTCNN or RetinaFace.

    Detects faces in images and returns bounding boxes, confidence scores,
    and optionally facial landmarks and cropped face images.

    Example:
        >>> config = ModelConfig(model_id="mtcnn", device="cuda")
        >>> detector = FaceDetector(config)
        >>> detector.load()
        >>> faces = detector.predict(image)
        >>> for face in faces:
        ...     print(f"Face at {face.bbox} with confidence {face.confidence}")
    """

    def __init__(
        self,
        config: ModelConfig,
        threshold: float = 0.9,
        min_face_size: int = 20,
        max_faces: int = 5,
        return_landmarks: bool = True,
        return_face_images: bool = True,
    ) -> None:
        """Initialize face detector.

        Args:
            config: Model configuration.
            threshold: Detection confidence threshold.
            min_face_size: Minimum face size in pixels.
            max_faces: Maximum number of faces to detect.
            return_landmarks: Whether to return facial landmarks.
            return_face_images: Whether to return cropped face images.
        """
        super().__init__(config)
        self.threshold = threshold
        self.min_face_size = min_face_size
        self.max_faces = max_faces
        self.return_landmarks = return_landmarks
        self.return_face_images = return_face_images

        self._detector: Any = None

    def load(self) -> None:
        """Load the face detection model."""
        if self._is_loaded:
            return

        model_type = self.config.model_id.lower()

        if model_type in ("mtcnn", "facenet-pytorch"):
            self._load_mtcnn()
        elif model_type == "retinaface":
            # RetinaFace would be loaded here
            # For now, fall back to MTCNN
            self._load_mtcnn()
        else:
            # Default to MTCNN
            self._load_mtcnn()

        self._is_loaded = True

    def _load_mtcnn(self) -> None:
        """Load MTCNN detector."""
        if not MTCNN_AVAILABLE:
            raise RuntimeError(
                "MTCNN not available. Install with: pip install facenet-pytorch"
            )

        import torch

        device = torch.device(self.config.device)
        self._detector = MTCNN(
            image_size=160,
            margin=14,
            min_face_size=self.min_face_size,
            thresholds=[0.6, 0.7, self.threshold],
            factor=0.709,
            post_process=False,
            device=device,
            keep_all=True,
        )

    def unload(self) -> None:
        """Unload the model."""
        self._detector = None
        self._is_loaded = False

    def predict(self, input_data: np.ndarray) -> list[FaceDetection]:
        """Detect faces in an image.

        Args:
            input_data: Input image as numpy array (H, W, C) in RGB format.

        Returns:
            List of FaceDetection objects.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if self._detector is None:
            return []

        # Ensure RGB format
        if len(input_data.shape) == 2:
            # Grayscale, convert to RGB
            input_data = np.stack([input_data] * 3, axis=-1)

        try:
            # MTCNN expects RGB images
            boxes, probs, landmarks = self._detector.detect(input_data, landmarks=True)
        except Exception:
            return []

        if boxes is None:
            return []

        detections = []
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob is None or prob < self.threshold:
                continue

            if len(detections) >= self.max_faces:
                break

            # Convert to integer coordinates
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(input_data.shape[1], x2)
            y2 = min(input_data.shape[0], y2)

            bbox = BoundingBox(
                x=x1,
                y=y1,
                width=x2 - x1,
                height=y2 - y1,
            )

            # Get landmarks if available
            face_landmarks = None
            if self.return_landmarks and landmarks is not None and i < len(landmarks):
                face_landmarks = landmarks[i]

            # Crop face image if requested
            face_image = None
            if self.return_face_images:
                face_image = input_data[y1:y2, x1:x2].copy()

            detections.append(
                FaceDetection(
                    bbox=bbox,
                    confidence=float(prob),
                    landmarks=face_landmarks,
                    face_image=face_image,
                )
            )

        return detections

    def detect_and_align(
        self,
        image: np.ndarray,
        target_size: tuple[int, int] = (224, 224),
    ) -> list[tuple[FaceDetection, np.ndarray]]:
        """Detect faces and return aligned face crops.

        Args:
            image: Input image in RGB format.
            target_size: Target size for aligned faces.

        Returns:
            List of (detection, aligned_face) tuples.
        """
        import cv2

        detections = self.predict(image)
        results = []

        for det in detections:
            if det.face_image is not None:
                aligned = cv2.resize(det.face_image, target_size)
                results.append((det, aligned))

        return results

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: list[FaceDetection],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw detection boxes on an image.

        Args:
            image: Input image.
            detections: List of face detections.
            color: Box color in BGR.
            thickness: Line thickness.

        Returns:
            Image with drawn boxes.
        """
        import cv2

        result = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox.to_xyxy()
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            # Draw confidence
            label = f"{det.confidence:.2f}"
            cv2.putText(
                result,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness,
            )

        return result

