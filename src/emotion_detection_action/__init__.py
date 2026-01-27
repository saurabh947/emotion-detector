"""
Emotion Detector SDK - Human emotion detection for robotics using VLA models.
"""

from emotion_detection_action.core.config import Config
from emotion_detection_action.core.detector import EmotionDetector
from emotion_detection_action.core.types import (
    ActionCommand,
    DetectionResult,
    EmotionResult,
    FaceDetection,
    VoiceDetection,
)

__version__ = "0.1.0"

__all__ = [
    "EmotionDetector",
    "Config",
    "EmotionResult",
    "DetectionResult",
    "ActionCommand",
    "FaceDetection",
    "VoiceDetection",
]

