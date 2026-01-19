# Emotion Detector SDK

Human emotion detection SDK for robotics using Vision-Language-Action (VLA) models.

## Features

- **Multi-modal emotion detection**: Analyze emotions from video, images, and audio
- **Face detection**: Automatic face detection using RetinaFace/MTCNN
- **Voice activity detection**: Detect human speech in audio streams
- **Facial emotion recognition**: Classify emotions from facial expressions
- **Speech emotion recognition**: Analyze emotions from voice/speech
- **Multimodal fusion**: Combine visual and audio emotion signals
- **Configurable VLA backend**: Swap VLA models (default: OpenVLA-7B)
- **Real-time & batch processing**: Support for both streaming and file-based inputs

## Installation

```bash
pip install emotion-detector
```

For VLA model support:

```bash
pip install emotion-detector[vla]
```

## Quick Start

### Batch Processing

```python
from emotion_detector import EmotionDetector, Config

config = Config(
    vla_model="openvla/openvla-7b",
    device="cuda"
)

detector = EmotionDetector(config)

# Process video and audio files
results = detector.process(
    video_path="recording.mp4",
    audio_path="recording.wav"
)

for result in results:
    print(f"Emotions: {result.emotions}")
    print(f"Action: {result.action}")
```

### Real-time Streaming

```python
import asyncio
from emotion_detector import EmotionDetector, Config

config = Config(
    vla_model="openvla/openvla-7b",
    device="cuda",
    mode="realtime"
)

detector = EmotionDetector(config)

async def main():
    async for result in detector.stream(video_source=0, audio_source=0):
        print(f"Emotions: {result.emotions}")
        print(f"Action: {result.action}")

asyncio.run(main())
```

## Architecture

```
Input Sources → Detection Layer → Emotion Analysis → VLA Model → Actions
     │                │                  │               │           │
  Video/Image    Face Detect      Facial Emotion    OpenVLA    (Stubbed)
  Audio          Voice Detect     Speech Emotion
                                  Fusion Module
```

## Configuration

```python
from emotion_detector import Config

config = Config(
    # VLA model configuration
    vla_model="openvla/openvla-7b",
    
    # Device settings
    device="cuda",  # or "cpu", "mps"
    
    # Processing mode
    mode="batch",  # or "realtime"
    
    # Detection thresholds
    face_detection_threshold=0.9,
    voice_activity_threshold=0.5,
    
    # Emotion model settings
    facial_emotion_model="trpakov/vit-face-expression",
    speech_emotion_model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
)
```

## Supported Emotions

- Happy
- Sad
- Angry
- Fearful
- Surprised
- Disgusted
- Neutral

## Development

```bash
# Clone the repository
git clone https://github.com/emotion-detector/emotion-detector.git
cd emotion-detector

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src tests
ruff check src tests --fix
```

## License

MIT License - see [LICENSE](LICENSE) for details.

