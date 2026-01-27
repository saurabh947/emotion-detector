"""Tests for input handlers."""

import numpy as np
import pytest

from emotion_detection_action.inputs.base import AudioChunk, VideoFrame


class TestVideoFrame:
    """Tests for VideoFrame container."""

    def test_creation(self):
        """Test creating a video frame."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = VideoFrame(data=data, timestamp=1.5, frame_number=45)

        assert frame.timestamp == 1.5
        assert frame.frame_number == 45

    def test_dimensions(self):
        """Test frame dimension properties."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = VideoFrame(data=data, timestamp=0.0, frame_number=0)

        assert frame.height == 480
        assert frame.width == 640
        assert frame.channels == 3

    def test_grayscale(self):
        """Test grayscale frame channels."""
        data = np.zeros((480, 640), dtype=np.uint8)
        frame = VideoFrame(data=data, timestamp=0.0, frame_number=0)

        assert frame.channels == 1


class TestAudioChunk:
    """Tests for AudioChunk container."""

    def test_creation(self):
        """Test creating an audio chunk."""
        data = np.zeros(16000, dtype=np.float32)
        chunk = AudioChunk(
            data=data,
            sample_rate=16000,
            start_time=1.0,
        )

        assert chunk.sample_rate == 16000
        assert chunk.start_time == 1.0
        assert chunk.channels == 1

    def test_duration(self):
        """Test duration calculation."""
        # 1 second of audio at 16kHz
        data = np.zeros(16000, dtype=np.float32)
        chunk = AudioChunk(data=data, sample_rate=16000, start_time=0.0)

        assert chunk.duration == 1.0

    def test_end_time(self):
        """Test end time calculation."""
        data = np.zeros(8000, dtype=np.float32)  # 0.5 seconds at 16kHz
        chunk = AudioChunk(data=data, sample_rate=16000, start_time=1.0)

        assert chunk.end_time == 1.5

    def test_num_samples(self):
        """Test number of samples."""
        data = np.zeros(12345, dtype=np.float32)
        chunk = AudioChunk(data=data, sample_rate=16000, start_time=0.0)

        assert chunk.num_samples == 12345


class TestVideoInput:
    """Tests for VideoInput handler."""

    def test_properties_before_open(self):
        """Test properties return defaults before opening."""
        from emotion_detection_action.inputs.video import VideoInput

        handler = VideoInput()
        assert handler.current_frame == 0
        assert handler.width == 0
        assert handler.height == 0
        assert not handler.is_open

    def test_close_idempotent(self):
        """Test that close can be called multiple times."""
        from emotion_detection_action.inputs.video import VideoInput

        handler = VideoInput()
        handler.close()  # Should not raise
        handler.close()  # Should not raise

    def test_read_before_open(self):
        """Test reading before opening returns None."""
        from emotion_detection_action.inputs.video import VideoInput

        handler = VideoInput()
        assert handler.read() is None

    def test_context_manager(self):
        """Test using handler as context manager."""
        from emotion_detection_action.inputs.video import VideoInput

        handler = VideoInput()
        with handler as h:
            assert h is handler

    def test_bgr_to_rgb(self):
        """Test BGR to RGB conversion."""
        from emotion_detection_action.inputs.video import VideoInput

        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # Blue channel
        
        rgb = VideoInput.bgr_to_rgb(bgr)
        
        assert rgb[:, :, 2].mean() == 255  # Red channel in RGB
        assert rgb[:, :, 0].mean() == 0  # Blue channel in RGB


class TestAudioInput:
    """Tests for AudioInput handler."""

    def test_properties_before_open(self):
        """Test properties return defaults before opening."""
        from emotion_detection_action.inputs.audio import AudioInput

        handler = AudioInput()
        assert handler.current_time == 0.0
        assert not handler.is_open

    def test_close_idempotent(self):
        """Test that close can be called multiple times."""
        from emotion_detection_action.inputs.audio import AudioInput

        handler = AudioInput()
        handler.close()  # Should not raise
        handler.close()  # Should not raise

    def test_read_before_open(self):
        """Test reading before opening returns None."""
        from emotion_detection_action.inputs.audio import AudioInput

        handler = AudioInput()
        assert handler.read() is None

    def test_context_manager(self):
        """Test using handler as context manager."""
        from emotion_detection_action.inputs.audio import AudioInput

        handler = AudioInput()
        with handler as h:
            assert h is handler

    def test_chunk_size_calculation(self):
        """Test chunk size is calculated from sample rate and duration."""
        from emotion_detection_action.inputs.audio import AudioInput

        handler = AudioInput(sample_rate=16000, chunk_duration=0.5)
        assert handler._chunk_size == 8000

        handler2 = AudioInput(sample_rate=44100, chunk_duration=1.0)
        assert handler2._chunk_size == 44100
