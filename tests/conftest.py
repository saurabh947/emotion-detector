"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest

from emotion_detection_action.core.types import (
    BoundingBox,
    EmotionScores,
    FaceDetection,
    VoiceDetection,
)


@pytest.fixture
def sample_frame():
    """Create a sample video frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_face_image():
    """Create a sample cropped face image."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_audio():
    """Create sample audio data (1 second at 16kHz)."""
    return np.random.randn(16000).astype(np.float32) * 0.1


@pytest.fixture
def sample_bbox():
    """Create a sample bounding box."""
    return BoundingBox(x=100, y=100, width=200, height=200)


@pytest.fixture
def sample_face_detection(sample_bbox, sample_face_image):
    """Create a sample face detection."""
    return FaceDetection(
        bbox=sample_bbox,
        confidence=0.95,
        face_image=sample_face_image,
    )


@pytest.fixture
def sample_voice_detection(sample_audio):
    """Create a sample voice detection."""
    return VoiceDetection(
        is_speech=True,
        confidence=0.9,
        start_time=0.0,
        end_time=1.0,
        audio_segment=sample_audio,
    )


@pytest.fixture
def sample_emotion_scores():
    """Create sample emotion scores."""
    return EmotionScores(
        happy=0.6,
        neutral=0.3,
        sad=0.1,
    )


@pytest.fixture
def happy_emotion_scores():
    """Create emotion scores for happy."""
    return EmotionScores(happy=0.9, neutral=0.1)


@pytest.fixture
def sad_emotion_scores():
    """Create emotion scores for sad."""
    return EmotionScores(sad=0.8, neutral=0.2)


@pytest.fixture
def angry_emotion_scores():
    """Create emotion scores for angry."""
    return EmotionScores(angry=0.85, neutral=0.15)


@pytest.fixture
def neutral_emotion_scores():
    """Create emotion scores for neutral."""
    return EmotionScores(neutral=0.9, happy=0.1)

