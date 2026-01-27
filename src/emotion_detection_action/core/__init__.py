"""Core module containing the main detector, configuration, and type definitions."""

from emotion_detection_action.core.config import Config
from emotion_detection_action.core.detector import EmotionDetector
from emotion_detection_action.core.types import (
    ActionCommand,
    DetectionResult,
    EmotionResult,
    FaceDetection,
    VoiceDetection,
)

__all__ = [
    "EmotionDetector",
    "Config",
    "EmotionResult",
    "DetectionResult",
    "ActionCommand",
    "FaceDetection",
    "VoiceDetection",
]

