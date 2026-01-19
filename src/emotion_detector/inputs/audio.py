"""Audio input handler for files and microphone streams."""

import asyncio
import queue
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import numpy as np
from scipy.io import wavfile

from emotion_detector.inputs.base import AudioChunk, BaseInput

# Try to import sounddevice, but make it optional
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False


class AudioInput(BaseInput[AudioChunk]):
    """Audio input handler supporting files and microphone streams.

    Handles reading from audio files (wav, mp3, etc.) and live microphone
    input using sounddevice.

    Example:
        >>> # From file
        >>> with AudioInput(sample_rate=16000) as audio:
        ...     audio.open("recording.wav")
        ...     for chunk in audio:
        ...         process(chunk)

        >>> # From microphone (real-time)
        >>> with AudioInput(sample_rate=16000) as audio:
        ...     audio.open_microphone(device=0)
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
        self._source: str | int | None = None
        self._audio_data: np.ndarray | None = None
        self._current_position = 0
        self._is_microphone = False

        # For microphone streaming
        self._audio_queue: queue.Queue[np.ndarray] | None = None
        self._stream: Any = None
        self._stop_event: threading.Event | None = None

    @property
    def duration(self) -> float:
        """Total duration in seconds (0 for live streams)."""
        if self._audio_data is None:
            return 0.0
        return len(self._audio_data) / self.sample_rate

    @property
    def current_time(self) -> float:
        """Current position in seconds."""
        return self._current_position / self.sample_rate

    def open(self, source: str | Path) -> None:
        """Open an audio file.

        Args:
            source: Path to audio file.

        Raises:
            ValueError: If file cannot be opened.
        """
        if self._is_open:
            self.close()

        source_path = Path(source)
        if not source_path.exists():
            raise ValueError(f"Audio file not found: {source}")

        self._source = str(source_path)
        self._is_microphone = False

        try:
            # Load audio file
            if source_path.suffix.lower() == ".wav":
                file_sr, audio = wavfile.read(str(source_path))
            else:
                # For other formats, try using scipy or fall back to basic wav
                # In production, you'd want to use librosa or audioread
                raise ValueError(f"Unsupported format: {source_path.suffix}. Use .wav files.")

            # Convert to float32 and normalize
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            elif audio.dtype == np.uint8:
                audio = (audio.astype(np.float32) - 128) / 128.0

            # Convert stereo to mono if needed
            if len(audio.shape) > 1 and self.channels == 1:
                audio = audio.mean(axis=1)

            # Resample if needed (basic linear interpolation)
            if file_sr != self.sample_rate:
                duration = len(audio) / file_sr
                new_length = int(duration * self.sample_rate)
                audio = np.interp(
                    np.linspace(0, len(audio), new_length),
                    np.arange(len(audio)),
                    audio,
                )

            self._audio_data = audio.astype(np.float32)
            self._current_position = 0
            self._is_open = True

        except Exception as e:
            raise ValueError(f"Could not load audio file: {e}")

    def open_microphone(self, device: int | str | None = None) -> None:
        """Open microphone for real-time streaming.

        Args:
            device: Audio device index or name. None for default.

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

        self._source = device
        self._is_microphone = True
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
        """Close the audio source."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._stop_event is not None:
            self._stop_event.set()
            self._stop_event = None

        self._audio_queue = None
        self._audio_data = None
        self._current_position = 0
        self._is_microphone = False
        self._is_open = False

    def read(self) -> AudioChunk | None:
        """Read the next audio chunk.

        Returns:
            AudioChunk or None if no more data.
        """
        if not self._is_open:
            return None

        if self._is_microphone:
            return self._read_microphone()
        else:
            return self._read_file()

    def _read_file(self) -> AudioChunk | None:
        """Read chunk from file."""
        if self._audio_data is None:
            return None

        if self._current_position >= len(self._audio_data):
            return None

        end_pos = min(
            self._current_position + self._chunk_size,
            len(self._audio_data),
        )
        chunk_data = self._audio_data[self._current_position:end_pos]

        # Pad last chunk if needed
        if len(chunk_data) < self._chunk_size:
            chunk_data = np.pad(
                chunk_data,
                (0, self._chunk_size - len(chunk_data)),
                mode="constant",
            )

        start_time = self._current_position / self.sample_rate
        self._current_position = end_pos

        return AudioChunk(
            data=chunk_data,
            sample_rate=self.sample_rate,
            start_time=start_time,
            channels=self.channels,
        )

    def _read_microphone(self) -> AudioChunk | None:
        """Read chunk from microphone."""
        if self._audio_queue is None:
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

    def seek(self, seconds: float) -> bool:
        """Seek to a specific time position.

        Args:
            seconds: Target time in seconds.

        Returns:
            True if seek was successful.
        """
        if self._is_microphone or self._audio_data is None:
            return False

        position = int(seconds * self.sample_rate)
        if 0 <= position < len(self._audio_data):
            self._current_position = position
            return True
        return False

    async def stream_async(self) -> AsyncIterator[AudioChunk]:
        """Async generator for streaming audio.

        Yields:
            Audio chunks from microphone or file.
        """
        while self._is_open:
            chunk = self.read()
            if chunk is None:
                if self._is_microphone:
                    await asyncio.sleep(0.01)
                    continue
                else:
                    break
            yield chunk

    @staticmethod
    def list_devices() -> list[dict[str, Any]]:
        """List available audio devices.

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
        mode = "microphone" if self._is_microphone else "file"
        return (
            f"AudioInput(source={self._source!r}, mode={mode}, "
            f"open={self._is_open}, time={self.current_time:.2f}s)"
        )

