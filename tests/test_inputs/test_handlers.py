"""Tests for input handlers."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from emotion_detection_action.inputs.base import AudioChunk, VideoFrame
from emotion_detection_action.inputs.image import ImageInput


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


class TestImageInput:
    """Tests for ImageInput handler."""

    def test_supported_extensions(self):
        """Test supported image extensions."""
        expected = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
        assert ImageInput.SUPPORTED_EXTENSIONS == expected

    def test_open_nonexistent_raises(self):
        """Test opening non-existent file raises error."""
        handler = ImageInput()
        with pytest.raises(ValueError, match="Source does not exist"):
            handler.open("/nonexistent/path/image.jpg")

    def test_open_unsupported_format_raises(self):
        """Test opening unsupported format raises error."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            temp_path = f.name

        try:
            handler = ImageInput()
            with pytest.raises(ValueError, match="Unsupported image format"):
                handler.open(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_open_empty_directory_raises(self):
        """Test opening empty directory raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ImageInput()
            with pytest.raises(ValueError, match="No supported images found"):
                handler.open(temp_dir)

    def test_properties_before_open(self):
        """Test properties return defaults before opening."""
        handler = ImageInput()
        assert handler.total_images == 0
        assert handler.current_index == 0
        assert not handler.is_open

    def test_close_idempotent(self):
        """Test that close can be called multiple times."""
        handler = ImageInput()
        handler.close()  # Should not raise
        handler.close()  # Should not raise

    def test_read_before_open(self):
        """Test reading before opening returns None."""
        handler = ImageInput()
        assert handler.read() is None

    def test_context_manager(self):
        """Test using handler as context manager."""
        handler = ImageInput()
        with handler as h:
            assert h is handler


class TestImageInputWithFiles:
    """Tests for ImageInput with actual files."""

    @pytest.fixture
    def temp_image(self):
        """Create a temporary test image."""
        import cv2

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        # Create a simple test image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(temp_path, img)

        yield temp_path

        Path(temp_path).unlink()

    @pytest.fixture
    def temp_image_dir(self):
        """Create a directory with test images."""
        import cv2

        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(3):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                cv2.imwrite(f"{temp_dir}/image_{i:02d}.png", img)

            yield temp_dir

    def test_open_single_image(self, temp_image):
        """Test opening a single image file."""
        handler = ImageInput()
        handler.open(temp_image)

        assert handler.is_open
        assert handler.total_images == 1

        handler.close()

    def test_read_single_image(self, temp_image):
        """Test reading a single image."""
        handler = ImageInput()
        handler.open(temp_image)

        frame = handler.read()

        assert frame is not None
        assert frame.data.shape == (100, 100, 3)
        assert frame.frame_number == 0

        # Second read should return None
        assert handler.read() is None

        handler.close()

    def test_open_directory(self, temp_image_dir):
        """Test opening a directory of images."""
        handler = ImageInput()
        handler.open(temp_image_dir)

        assert handler.is_open
        assert handler.total_images == 3

        handler.close()

    def test_iterate_directory(self, temp_image_dir):
        """Test iterating through directory images."""
        handler = ImageInput()
        handler.open(temp_image_dir)

        frames = list(handler)

        assert len(frames) == 3
        for i, frame in enumerate(frames):
            assert frame.frame_number == i

        handler.close()

    def test_seek(self, temp_image_dir):
        """Test seeking to specific image."""
        handler = ImageInput()
        handler.open(temp_image_dir)

        assert handler.seek(2) is True
        frame = handler.read()
        assert frame is not None
        assert frame.frame_number == 2

        # Seek out of bounds
        assert handler.seek(100) is False

        handler.close()

    def test_get_image_at(self, temp_image_dir):
        """Test getting image at specific index."""
        handler = ImageInput()
        handler.open(temp_image_dir)

        frame = handler.get_image_at(1)
        assert frame is not None
        assert frame.frame_number == 1

        # Original position should be unchanged
        next_frame = handler.read()
        assert next_frame is not None
        assert next_frame.frame_number == 0

        handler.close()

    def test_get_paths(self, temp_image_dir):
        """Test getting list of image paths."""
        handler = ImageInput()
        handler.open(temp_image_dir)

        paths = handler.get_paths()

        assert len(paths) == 3
        assert all(p.suffix == ".png" for p in paths)

        handler.close()

    def test_load_single_static(self, temp_image):
        """Test static method for loading single image."""
        img = ImageInput.load_single(temp_image)

        assert img is not None
        assert img.shape == (100, 100, 3)

