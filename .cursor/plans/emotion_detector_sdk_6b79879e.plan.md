---
name: Emotion Detector SDK
overview: Design and build an open-source Python SDK that detects human emotions from video/image/audio inputs using specialized emotion recognition models, then feeds the emotional context to a configurable VLA model (defaulting to OpenVLA-7B) for robot action generation.
todos:
  - id: scaffold
    content: Create project structure with pyproject.toml, README, and package layout
    status: completed
  - id: core-types
    content: Implement core types, config system, and model registry pattern
    status: completed
    dependencies:
      - scaffold
  - id: input-handlers
    content: Build video, image, and audio input handlers with async support
    status: completed
    dependencies:
      - core-types
  - id: face-detection
    content: Implement face detection using RetinaFace/MTCNN
    status: completed
    dependencies:
      - input-handlers
  - id: voice-detection
    content: Implement voice activity detection for audio streams
    status: completed
    dependencies:
      - input-handlers
  - id: facial-emotion
    content: Implement facial emotion recognition with ViT-based model
    status: completed
    dependencies:
      - face-detection
  - id: speech-emotion
    content: Implement speech emotion recognition with Wav2Vec2
    status: completed
    dependencies:
      - voice-detection
  - id: fusion
    content: Create multimodal emotion fusion module
    status: completed
    dependencies:
      - facial-emotion
      - speech-emotion
  - id: vla-integration
    content: Integrate OpenVLA with configurable model loading
    status: completed
    dependencies:
      - fusion
  - id: action-stub
    content: Create stubbed action handler interface
    status: completed
    dependencies:
      - vla-integration
  - id: main-detector
    content: Build main EmotionDetector class orchestrating the pipeline
    status: completed
    dependencies:
      - action-stub
  - id: examples
    content: Create usage examples for real-time and batch processing
    status: completed
    dependencies:
      - main-detector
---

# Emotion Detector SDK for Robotics

## Architecture Overview

```mermaid
flowchart LR
    subgraph inputs [Input Sources]
        Video[Video Stream]
        Image[Images]
        Audio[Audio Stream]
    end
    
    subgraph detection [Detection Layer]
        FaceDetect[Face Detector]
        VoiceDetect[Voice Activity Detector]
    end
    
    subgraph emotion [Emotion Analysis]
        FER[Facial Emotion Recognition]
        SER[Speech Emotion Recognition]
        Fusion[Multimodal Fusion]
    end
    
    subgraph action [Action Layer]
        VLA[VLA Model - OpenVLA]
        ActionStub[Action Output - Stubbed]
    end
    
    Video --> FaceDetect
    Image --> FaceDetect
    Audio --> VoiceDetect
    FaceDetect --> FER
    VoiceDetect --> SER
    FER --> Fusion
    SER --> Fusion
    Fusion --> VLA
    VLA --> ActionStub
```



## Model Selection

| Component | Recommended Model | HuggingFace Path ||-----------|------------------|------------------|| VLA (configurable) | OpenVLA-7B | `openvla/openvla-7b` || Face Detection | RetinaFace / MTCNN | `timesler/facenet-pytorch` || Facial Emotion | FER2013-based models | `trpakov/vit-face-expression` || Speech Emotion | Wav2Vec2-based | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` |

## Proposed Folder Structure

```javascript
emotion-detector/
├── src/
│   └── emotion_detector/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── detector.py          # Main EmotionDetector class
│       │   ├── config.py            # Configuration management
│       │   └── types.py             # Type definitions, dataclasses
│       ├── inputs/
│       │   ├── __init__.py
│       │   ├── base.py              # Abstract input handler
│       │   ├── video.py             # Video/webcam processing
│       │   ├── image.py             # Image processing
│       │   └── audio.py             # Audio/microphone processing
│       ├── detection/
│       │   ├── __init__.py
│       │   ├── face.py              # Face detection
│       │   └── voice.py             # Voice activity detection
│       ├── emotion/
│       │   ├── __init__.py
│       │   ├── facial.py            # Facial emotion recognition
│       │   ├── speech.py            # Speech emotion recognition
│       │   └── fusion.py            # Multimodal emotion fusion
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py              # Abstract model interface
│       │   ├── registry.py          # Model registry for configurability
│       │   └── vla/
│       │       ├── __init__.py
│       │       ├── base.py          # VLA interface
│       │       └── openvla.py       # OpenVLA implementation
│       └── actions/
│           ├── __init__.py
│           ├── base.py              # Action interface
│           └── stub.py              # Stubbed action handler
├── tests/
│   ├── __init__.py
│   ├── test_inputs/
│   ├── test_detection/
│   ├── test_emotion/
│   └── test_integration/
├── examples/
│   ├── basic_usage.py
│   ├── realtime_webcam.py
│   └── batch_processing.py
├── pyproject.toml
├── README.md
└── LICENSE
```



## Key Design Decisions

1. **Model Registry Pattern**: All models (VLA, emotion, detection) registered via a central registry, allowing runtime configuration
2. **Abstract Base Classes**: Each component has an interface, enabling custom implementations
3. **Async Support**: Real-time processing uses async/await for non-blocking I/O
4. **Dataclasses for Types**: Strong typing with `EmotionResult`, `DetectionResult`, `ActionCommand`

## SDK API Design (Target Interface)

```python
from emotion_detector import EmotionDetector, Config

# Configure with custom VLA model
config = Config(
    vla_model="openvla/openvla-7b",  # Configurable
    device="cuda",
    mode="realtime"  # or "batch"
)

detector = EmotionDetector(config)

# Real-time mode
async for result in detector.stream(video_source=0, audio_source=0):
    print(result.emotions)  # {"happy": 0.8, "neutral": 0.2}
    print(result.action)    # Stubbed action output

# Batch mode
results = detector.process(
    video_path="recording.mp4",
    audio_path="recording.wav"
)



```