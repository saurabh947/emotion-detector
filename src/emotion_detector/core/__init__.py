"""Core module containing the main detector, configuration, and type definitions."""

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

