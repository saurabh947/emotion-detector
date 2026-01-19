"""
Emotion Detector SDK - Human emotion detection for robotics using VLA models.
"""

from emotion_detector.core.config import Config
from emotion_detector.core.detector import EmotionDetector
from emotion_detector.core.types import (
    ActionCommand,
    DetectionResult,
    EmotionResult,
    FaceDetection,
    ProcessingMode,
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
    "ProcessingMode",
]

