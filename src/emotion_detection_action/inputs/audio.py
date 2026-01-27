"""Audio input handler for real-time microphone streams."""

import asyncio
import queue
import threading
from typing import Any, AsyncIterator

import numpy as np

from emotion_detection_action.inputs.base import AudioChunk, BaseInput

# Try to import sounddevice, but make it optional
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False


class AudioInput(BaseInput[AudioChunk]):
    """Audio input handler for real-time microphone streams.

    Handles reading from live microphone input using sounddevice.

    Example:
        >>> with AudioInput(sample_rate=16000) as audio:
        ...     audio.open(device=0)  # or None for default
        ...     for chunk in audio:
        ...         process(chunk)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,
        channels: int = 1,
    ) -> None:
        """Initialize audio input handler.

        Args:
            sample_rate: Target sample rate in Hz.
            chunk_duration: Duration of each audio chunk in seconds.
            channels: Number of audio channels (1=mono, 2=stereo).
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.channels = channels

        self._chunk_size = int(sample_rate * chunk_duration)
        self._device: int | str | None = None
        self._current_position = 0

        # For microphone streaming
        self._audio_queue: queue.Queue[np.ndarray] | None = None
        self._stream: Any = None
        self._stop_event: threading.Event | None = None

    @property
    def current_time(self) -> float:
        """Current position in seconds."""
        return self._current_position / self.sample_rate

    def open(self, device: int | str | None = None) -> None:
        """Open microphone for real-time streaming.

        Args:
            device: Audio device index or name. None for default device.

        Raises:
            RuntimeError: If sounddevice is not available.
            ValueError: If device cannot be opened.
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError(
                "sounddevice is not available. Install it with: pip install sounddevice"
            )

        if self._is_open:
            self.close()

        self._device = device
        self._audio_queue = queue.Queue()
        self._stop_event = threading.Event()

        def audio_callback(
            indata: np.ndarray,
            frames: int,
            time_info: Any,
            status: Any,
        ) -> None:
            if status:
                print(f"Audio status: {status}")
            if self._audio_queue is not None:
                self._audio_queue.put(indata.copy())

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self._chunk_size,
                device=device,
                callback=audio_callback,
            )
            self._stream.start()
            self._is_open = True
        except Exception as e:
            self._audio_queue = None
            self._stop_event = None
            raise ValueError(f"Could not open microphone: {e}")

    def close(self) -> None:
        """Close the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._stop_event is not None:
            self._stop_event.set()
            self._stop_event = None

        self._audio_queue = None
        self._current_position = 0
        self._is_open = False

    def read(self) -> AudioChunk | None:
        """Read the next audio chunk from microphone.

        Returns:
            AudioChunk or None if no data available.
        """
        if not self._is_open or self._audio_queue is None:
            return None

        try:
            # Block with timeout
            data = self._audio_queue.get(timeout=1.0)
            if self.channels == 1 and len(data.shape) > 1:
                data = data.mean(axis=1)

            start_time = self._current_position / self.sample_rate
            self._current_position += len(data)

            return AudioChunk(
                data=data.flatten().astype(np.float32),
                sample_rate=self.sample_rate,
                start_time=start_time,
                channels=self.channels,
            )
        except queue.Empty:
            return None

    async def stream_async(self) -> AsyncIterator[AudioChunk]:
        """Async generator for streaming audio.

        Yields:
            Audio chunks from microphone.
        """
        while self._is_open:
            chunk = self.read()
            if chunk is None:
                await asyncio.sleep(0.01)
                continue
            yield chunk

    @staticmethod
    def list_devices() -> list[dict[str, Any]]:
        """List available audio input devices.

        Returns:
            List of device info dictionaries.
        """
        if not SOUNDDEVICE_AVAILABLE:
            return []
        return [
            {
                "index": i,
                "name": d["name"],
                "channels": d["max_input_channels"],
                "sample_rate": d["default_samplerate"],
            }
            for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0
        ]

    def __repr__(self) -> str:
        return (
            f"AudioInput(device={self._device!r}, "
            f"open={self._is_open}, time={self.current_time:.2f}s)"
        )
