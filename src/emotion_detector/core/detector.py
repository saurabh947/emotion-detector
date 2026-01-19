"""Main EmotionDetector class orchestrating the full pipeline."""

import asyncio
from typing import AsyncIterator

import numpy as np

from emotion_detector.actions.base import BaseActionHandler
from emotion_detector.actions.logging_handler import LoggingActionHandler
from emotion_detector.core.config import Config, ModelConfig
from emotion_detector.core.types import (
    ActionCommand,
    DetectionResult,
    EmotionResult,
    PipelineResult,
    ProcessingMode,
)
from emotion_detector.detection.face import FaceDetector
from emotion_detector.detection.voice import VoiceActivityDetector
from emotion_detector.emotion.facial import FacialEmotionRecognizer
from emotion_detector.emotion.fusion import EmotionFusion
from emotion_detector.emotion.speech import SpeechEmotionRecognizer
from emotion_detector.inputs.audio import AudioInput
from emotion_detector.inputs.base import AudioChunk, VideoFrame
from emotion_detector.inputs.image import ImageInput
from emotion_detector.inputs.video import VideoInput
from emotion_detector.models.vla.base import BaseVLAModel, VLAInput
from emotion_detector.models.vla.openvla import OpenVLAModel


class EmotionDetector:
    """Main emotion detector class orchestrating the full pipeline.

    This class integrates all components of the emotion detection system:
    - Input handlers (video, image, audio)
    - Face detection
    - Voice activity detection
    - Facial emotion recognition
    - Speech emotion recognition
    - Multimodal fusion
    - VLA-based action generation

    Example:
        >>> from emotion_detector import EmotionDetector, Config

        >>> # Create detector with default config
        >>> detector = EmotionDetector()
        >>> detector.initialize()

        >>> # Process a video file
        >>> results = detector.process(video_path="video.mp4")
        >>> for result in results:
        ...     print(result.emotion.dominant_emotion)

        >>> # Real-time processing
        >>> async for result in detector.stream(video_source=0):
        ...     print(result.emotion.dominant_emotion)
    """

    def __init__(
        self,
        config: Config | None = None,
        action_handler: BaseActionHandler | None = None,
    ) -> None:
        """Initialize the emotion detector.

        Args:
            config: Configuration for the detector. Uses defaults if None.
            action_handler: Custom action handler. Uses stub if None.
        """
        self.config = config or Config()
        self.action_handler = action_handler or LoggingActionHandler(verbose=self.config.verbose)

        # Components (initialized lazily)
        self._face_detector: FaceDetector | None = None
        self._voice_detector: VoiceActivityDetector | None = None
        self._facial_emotion: FacialEmotionRecognizer | None = None
        self._speech_emotion: SpeechEmotionRecognizer | None = None
        self._fusion: EmotionFusion | None = None
        self._vla_model: BaseVLAModel | None = None

        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if detector is initialized."""
        return self._initialized

    def initialize(self, load_vla: bool = True) -> None:
        """Initialize all pipeline components.

        Args:
            load_vla: Whether to load the VLA model. Set False for emotion-only mode.
        """
        if self._initialized:
            return

        # Initialize face detector
        face_config = ModelConfig(
            model_id=self.config.face_detection_model,
            device=self.config.device,
            extra_kwargs={"threshold": self.config.face_detection_threshold},
        )
        self._face_detector = FaceDetector(
            face_config,
            threshold=self.config.face_detection_threshold,
            min_face_size=self.config.face_min_size,
            max_faces=self.config.max_faces,
        )
        self._face_detector.load()

        # Initialize voice activity detector
        vad_config = ModelConfig(model_id="webrtcvad", device="cpu")
        self._voice_detector = VoiceActivityDetector(
            vad_config,
            aggressiveness=self.config.vad_aggressiveness,
        )
        self._voice_detector.load()

        # Initialize facial emotion recognizer
        facial_config = self.config.get_facial_emotion_config()
        self._facial_emotion = FacialEmotionRecognizer(facial_config)
        self._facial_emotion.load()

        # Initialize speech emotion recognizer
        speech_config = self.config.get_speech_emotion_config()
        self._speech_emotion = SpeechEmotionRecognizer(
            speech_config,
            target_sample_rate=self.config.sample_rate,
        )
        self._speech_emotion.load()

        # Initialize fusion module
        self._fusion = EmotionFusion(
            strategy=self.config.fusion_strategy,
            visual_weight=self.config.facial_weight,
            audio_weight=self.config.speech_weight,
        )

        # Initialize VLA model (optional)
        if load_vla and self.config.vla_enabled:
            vla_config = self.config.get_vla_config()
            self._vla_model = OpenVLAModel(vla_config)
            self._vla_model.load()

        # Connect action handler
        self.action_handler.connect()

        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown and release all resources."""
        if self._face_detector:
            self._face_detector.unload()
        if self._voice_detector:
            self._voice_detector.unload()
        if self._facial_emotion:
            self._facial_emotion.unload()
        if self._speech_emotion:
            self._speech_emotion.unload()
        if self._vla_model:
            self._vla_model.unload()

        self.action_handler.disconnect()
        self._initialized = False

    def process(
        self,
        video_path: str | None = None,
        image_path: str | None = None,
        audio_path: str | None = None,
    ) -> list[PipelineResult]:
        """Process video/image/audio files in batch mode.

        Args:
            video_path: Path to video file.
            image_path: Path to image file or directory.
            audio_path: Path to audio file.

        Returns:
            List of pipeline results for each processed frame.
        """
        if not self._initialized:
            self.initialize()

        results: list[PipelineResult] = []

        # Process image(s)
        if image_path:
            image_input = ImageInput()
            image_input.open(image_path)
            for frame in image_input:
                result = self._process_frame(frame, None)
                if result:
                    results.append(result)
            image_input.close()
            return results

        # Process video with optional audio
        if video_path:
            video_input = VideoInput(frame_skip=self.config.frame_skip)
            video_input.open(video_path)

            audio_chunks: list[AudioChunk] = []
            if audio_path:
                audio_input = AudioInput(sample_rate=self.config.sample_rate)
                audio_input.open(audio_path)
                audio_chunks = list(audio_input)
                audio_input.close()

            frame_idx = 0
            for frame in video_input:
                # Get corresponding audio chunk if available
                audio_chunk = None
                if audio_chunks:
                    chunk_idx = int(frame.timestamp / 0.5)  # Assuming 0.5s chunks
                    if chunk_idx < len(audio_chunks):
                        audio_chunk = audio_chunks[chunk_idx]

                result = self._process_frame(frame, audio_chunk)
                if result:
                    results.append(result)

                frame_idx += 1

            video_input.close()

        # Process audio only
        elif audio_path:
            audio_input = AudioInput(sample_rate=self.config.sample_rate)
            audio_input.open(audio_path)
            for chunk in audio_input:
                result = self._process_audio_only(chunk)
                if result:
                    results.append(result)
            audio_input.close()

        return results

    async def stream(
        self,
        video_source: int | str = 0,
        audio_source: int | str | None = None,
    ) -> AsyncIterator[PipelineResult]:
        """Stream real-time emotion detection.

        Args:
            video_source: Video source (camera index or URL).
            audio_source: Audio source (device index). None to disable audio.

        Yields:
            Pipeline results for each processed frame.
        """
        if not self._initialized:
            self.initialize()

        video_input = VideoInput(frame_skip=self.config.frame_skip)
        video_input.open(video_source)

        audio_input: AudioInput | None = None
        if audio_source is not None:
            audio_input = AudioInput(sample_rate=self.config.sample_rate)
            audio_input.open_microphone(audio_source)

        try:
            async for frame in video_input:
                # Get audio chunk if available
                audio_chunk = None
                if audio_input:
                    audio_chunk = audio_input.read()

                result = self._process_frame(frame, audio_chunk)
                if result:
                    yield result

                # Small delay to prevent CPU overload
                await asyncio.sleep(0.001)

        finally:
            video_input.close()
            if audio_input:
                audio_input.close()

    def process_frame(
        self,
        frame: np.ndarray,
        audio: np.ndarray | None = None,
        timestamp: float = 0.0,
    ) -> PipelineResult | None:
        """Process a single frame with optional audio.

        Lower-level API for custom processing pipelines.

        Args:
            frame: Video frame as numpy array (H, W, C) in BGR format.
            audio: Optional audio data as numpy array.
            timestamp: Frame timestamp in seconds.

        Returns:
            Pipeline result or None if processing failed.
        """
        if not self._initialized:
            self.initialize()

        video_frame = VideoFrame(data=frame, timestamp=timestamp, frame_number=0)

        audio_chunk = None
        if audio is not None:
            audio_chunk = AudioChunk(
                data=audio,
                sample_rate=self.config.sample_rate,
                start_time=timestamp,
            )

        return self._process_frame(video_frame, audio_chunk)

    def _process_frame(
        self,
        frame: VideoFrame,
        audio_chunk: AudioChunk | None,
    ) -> PipelineResult | None:
        """Internal method to process a frame through the pipeline.

        Args:
            frame: Video frame to process.
            audio_chunk: Optional audio chunk.

        Returns:
            Pipeline result or None.
        """
        assert self._face_detector is not None
        assert self._facial_emotion is not None
        assert self._fusion is not None

        timestamp = frame.timestamp

        # Convert BGR to RGB for face detection
        rgb_frame = VideoInput.bgr_to_rgb(frame.data)

        # Detect faces
        faces = self._face_detector.predict(rgb_frame)

        # Process voice activity if audio available
        voice_detection = None
        if audio_chunk and self._voice_detector:
            voice_detection = self._voice_detector.predict(audio_chunk)

        # Create detection result
        detection = DetectionResult(
            timestamp=timestamp,
            faces=faces,
            voice=voice_detection,
            frame=frame.data,
        )

        # Facial emotion recognition
        facial_result = None
        if faces:
            # Process first face (could extend to multiple)
            facial_result = self._facial_emotion.predict(faces[0])

        # Speech emotion recognition
        speech_result = None
        if voice_detection and voice_detection.is_speech and self._speech_emotion:
            speech_result = self._speech_emotion.predict(voice_detection)

        # Fuse emotions
        if facial_result is None and speech_result is None:
            return None

        emotion_result = self._fusion.fuse(facial_result, speech_result, timestamp)

        # Generate action
        action = self._generate_action(emotion_result, frame.data)

        # Execute action through handler
        self.action_handler.execute(action)

        return PipelineResult(
            timestamp=timestamp,
            detection=detection,
            emotion=emotion_result,
            action=action,
        )

    def _process_audio_only(self, audio_chunk: AudioChunk) -> PipelineResult | None:
        """Process audio-only input.

        Args:
            audio_chunk: Audio chunk to process.

        Returns:
            Pipeline result or None.
        """
        assert self._voice_detector is not None
        assert self._speech_emotion is not None
        assert self._fusion is not None

        # Detect voice activity
        voice_detection = self._voice_detector.predict(audio_chunk)

        if not voice_detection or not voice_detection.is_speech:
            return None

        # Speech emotion recognition
        speech_result = self._speech_emotion.predict(voice_detection)

        # Create detection result (no visual data)
        detection = DetectionResult(
            timestamp=audio_chunk.start_time,
            faces=[],
            voice=voice_detection,
            frame=None,
        )

        # Fuse (speech only)
        emotion_result = self._fusion.fuse(None, speech_result, audio_chunk.start_time)

        # Generate action
        action = self._generate_action(emotion_result, None)

        # Execute action
        self.action_handler.execute(action)

        return PipelineResult(
            timestamp=audio_chunk.start_time,
            detection=detection,
            emotion=emotion_result,
            action=action,
        )

    def _generate_action(
        self,
        emotion: EmotionResult,
        image: np.ndarray | None,
    ) -> ActionCommand:
        """Generate action based on emotion.

        Args:
            emotion: Detected emotion result.
            image: Optional visual context.

        Returns:
            Generated action command.
        """
        # Use VLA model if available
        if self._vla_model and self._vla_model.is_loaded:
            return self._vla_model.generate_emotion_response(emotion, image)

        # Fall back to stub action
        return ActionCommand.stub(emotion)

    def get_emotion_only(
        self,
        frame: np.ndarray,
        audio: np.ndarray | None = None,
    ) -> EmotionResult | None:
        """Get emotion result without action generation.

        Useful when you only need emotion detection.

        Args:
            frame: Video frame in BGR format.
            audio: Optional audio data.

        Returns:
            Emotion result or None.
        """
        result = self.process_frame(frame, audio)
        return result.emotion if result else None

    def __enter__(self) -> "EmotionDetector":
        """Context manager entry - initialize."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - shutdown."""
        self.shutdown()

    def __repr__(self) -> str:
        return (
            f"EmotionDetector(mode={self.config.mode.value}, "
            f"vla={self.config.vla_model}, initialized={self._initialized})"
        )
