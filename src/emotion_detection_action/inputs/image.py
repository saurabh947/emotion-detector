"""Image input handler for single images and image directories."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from emotion_detection_action.inputs.base import BaseInput, VideoFrame


class ImageInput(BaseInput[VideoFrame]):
    """Image input handler for single images and directories.

    Supports common image formats (jpg, png, bmp, etc.) and can process
    single images or iterate through a directory of images.

    Example:
        >>> # Single image
        >>> with ImageInput() as img:
        ...     img.open("photo.jpg")
        ...     frame = img.read()
        ...     process(frame)

        >>> # Directory of images
        >>> with ImageInput() as img:
        ...     img.open("images/")
        ...     for frame in img:
        ...         process(frame)
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

    def __init__(self) -> None:
        """Initialize image input handler."""
        super().__init__()
        self._source: Path | None = None
        self._images: list[Path] = []
        self._current_index = 0
        self._single_image_read = False

    @property
    def total_images(self) -> int:
        """Total number of images."""
        return len(self._images)

    @property
    def current_index(self) -> int:
        """Current image index."""
        return self._current_index

    def open(self, source: str | Path) -> None:
        """Open an image source.

        Args:
            source: Image file path or directory containing images.

        Raises:
            ValueError: If source doesn't exist or has no valid images.
        """
        if self._is_open:
            self.close()

        self._source = Path(source)

        if not self._source.exists():
            raise ValueError(f"Source does not exist: {source}")

        if self._source.is_file():
            if self._source.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported image format: {self._source.suffix}")
            self._images = [self._source]
        elif self._source.is_dir():
            self._images = sorted(
                p for p in self._source.iterdir()
                if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
            )
            if not self._images:
                raise ValueError(f"No supported images found in: {source}")
        else:
            raise ValueError(f"Invalid source: {source}")

        self._current_index = 0
        self._single_image_read = False
        self._is_open = True

    def close(self) -> None:
        """Close the image source."""
        self._images = []
        self._current_index = 0
        self._single_image_read = False
        self._is_open = False

    def read(self) -> VideoFrame | None:
        """Read the next image.

        Returns:
            VideoFrame containing the image, or None if no more images.
        """
        if not self._is_open:
            return None

        if self._current_index >= len(self._images):
            return None

        image_path = self._images[self._current_index]
        image = self._load_image(image_path)

        if image is None:
            # Skip corrupted images
            self._current_index += 1
            return self.read()

        frame = VideoFrame(
            data=image,
            timestamp=float(self._current_index),
            frame_number=self._current_index,
        )
        self._current_index += 1
        return frame

    def _load_image(self, path: Path) -> np.ndarray | None:
        """Load an image from file.

        Args:
            path: Path to image file.

        Returns:
            Image as numpy array (BGR) or None if load fails.
        """
        try:
            # Use OpenCV for consistency with video frames
            image = cv2.imread(str(path))
            if image is None:
                # Fallback to PIL for formats OpenCV doesn't handle well
                pil_image = Image.open(path).convert("RGB")
                image = np.array(pil_image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        except Exception:
            return None

    def get_image_at(self, index: int) -> VideoFrame | None:
        """Get a specific image by index.

        Args:
            index: Image index.

        Returns:
            VideoFrame or None if index is invalid.
        """
        if not self._is_open or index < 0 or index >= len(self._images):
            return None

        image_path = self._images[index]
        image = self._load_image(image_path)

        if image is None:
            return None

        return VideoFrame(
            data=image,
            timestamp=float(index),
            frame_number=index,
        )

    def seek(self, index: int) -> bool:
        """Seek to a specific image index.

        Args:
            index: Target index.

        Returns:
            True if seek was successful.
        """
        if 0 <= index < len(self._images):
            self._current_index = index
            return True
        return False

    def get_paths(self) -> list[Path]:
        """Get list of all image paths.

        Returns:
            List of image file paths.
        """
        return self._images.copy()

    @staticmethod
    def load_single(path: str | Path) -> np.ndarray | None:
        """Load a single image without using the iterator interface.

        Args:
            path: Path to image file.

        Returns:
            Image as numpy array (BGR) or None if load fails.
        """
        handler = ImageInput()
        handler.open(path)
        frame = handler.read()
        handler.close()
        return frame.data if frame else None

    def __repr__(self) -> str:
        return (
            f"ImageInput(source={self._source!r}, open={self._is_open}, "
            f"images={self._current_index}/{len(self._images)})"
        )

