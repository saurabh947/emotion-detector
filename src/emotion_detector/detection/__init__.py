"""Detection modules for face and voice activity detection."""

from emotion_detector.detection.face import FaceDetector
from emotion_detector.detection.voice import VoiceActivityDetector

__all__ = [
    "FaceDetector",
    "VoiceActivityDetector",
]

