"""Emotion recognition modules for facial and speech emotion analysis."""

from emotion_detector.emotion.facial import FacialEmotionRecognizer
from emotion_detector.emotion.fusion import EmotionFusion
from emotion_detector.emotion.speech import SpeechEmotionRecognizer

__all__ = [
    "FacialEmotionRecognizer",
    "SpeechEmotionRecognizer",
    "EmotionFusion",
]

