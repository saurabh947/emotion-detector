"""Video input handler for files and camera streams."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from emotion_detection_action.inputs.base import BaseInput, VideoFrame


class VideoInput(BaseInput[VideoFrame]):
    """Video input handler supporting files and camera streams.

    Handles reading from video files (mp4, avi, etc.) and live camera
    feeds using OpenCV.

    Example:
        >>> with VideoInput() as video:
        ...     video.open("recording.mp4")
        ...     for frame in video:
        ...         process(frame)

        >>> # Or with camera
        >>> with VideoInput() as video:
        ...     video.open(0)  # Camera index
        ...     for frame in video:
        ...         process(frame)
    """

    def __init__(self, frame_skip: int = 1) -> None:
        """Initialize video input handler.

        Args:
            frame_skip: Process every nth frame (1 = all frames).
        """
        super().__init__()
        self._cap: cv2.VideoCapture | None = None
        self._frame_number = 0
        self._frame_skip = max(1, frame_skip)
        self._fps: float = 30.0
        self._total_frames: int = 0
        self._source: str | int | None = None

    @property
    def fps(self) -> float:
        """Frames per second of the video."""
        return self._fps

    @property
    def total_frames(self) -> int:
        """Total number of frames (0 for live streams)."""
        return self._total_frames

    @property
    def current_frame(self) -> int:
        """Current frame number."""
        return self._frame_number

    @property
    def width(self) -> int:
        """Frame width."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Frame height."""
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def open(self, source: str | int | Path) -> None:
        """Open a video source.

        Args:
            source: Video file path or camera index.

        Raises:
            ValueError: If source cannot be opened.
        """
        if self._is_open:
            self.close()

        if isinstance(source, Path):
            source = str(source)

        self._source = source
        self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_number = 0
        self._is_open = True

    def close(self) -> None:
        """Close the video source."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False
        self._frame_number = 0

    def read(self) -> VideoFrame | None:
        """Read the next frame.

        Returns:
            VideoFrame or None if no more frames.
        """
        if not self._is_open or self._cap is None:
            return None

        # Skip frames if needed
        for _ in range(self._frame_skip - 1):
            self._cap.read()
            self._frame_number += 1

        ret, frame = self._cap.read()
        if not ret:
            return None

        timestamp = self._frame_number / self._fps
        result = VideoFrame(
            data=frame,
            timestamp=timestamp,
            frame_number=self._frame_number,
        )
        self._frame_number += 1
        return result

    def seek(self, frame_number: int) -> bool:
        """Seek to a specific frame.

        Args:
            frame_number: Target frame number.

        Returns:
            True if seek was successful.
        """
        if self._cap is None:
            return False

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self._frame_number = frame_number
        return True

    def seek_time(self, seconds: float) -> bool:
        """Seek to a specific time.

        Args:
            seconds: Target time in seconds.

        Returns:
            True if seek was successful.
        """
        frame_number = int(seconds * self._fps)
        return self.seek(frame_number)

    def get_frame_at(self, frame_number: int) -> VideoFrame | None:
        """Get a specific frame without advancing position.

        Args:
            frame_number: Frame number to retrieve.

        Returns:
            VideoFrame or None if not available.
        """
        if self._cap is None:
            return None

        current_pos = self._frame_number
        self.seek(frame_number)
        frame = self.read()
        self.seek(current_pos)
        return frame

    @staticmethod
    def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to RGB.

        Args:
            frame: BGR image array.

        Returns:
            RGB image array.
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __repr__(self) -> str:
        return (
            f"VideoInput(source={self._source!r}, open={self._is_open}, "
            f"frame={self._frame_number}/{self._total_frames})"
        )

