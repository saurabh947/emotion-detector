"""Emotion recognition modules for facial and speech emotion analysis."""

from emotion_detection_action.emotion.facial import FacialEmotionRecognizer
from emotion_detection_action.emotion.fusion import EmotionFusion
from emotion_detection_action.emotion.speech import SpeechEmotionRecognizer

__all__ = [
    "FacialEmotionRecognizer",
    "SpeechEmotionRecognizer",
    "EmotionFusion",
]

