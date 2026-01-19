"""Multimodal emotion fusion module."""

from typing import Literal

import numpy as np

from emotion_detector.core.types import (
    EmotionResult,
    EmotionScores,
    FacialEmotionResult,
    SpeechEmotionResult,
)


class EmotionFusion:
    """Fuses facial and speech emotion recognition results.

    Combines emotion predictions from multiple modalities (visual and audio)
    into a unified emotion result using various fusion strategies.

    Example:
        >>> fusion = EmotionFusion(strategy="confidence", confidence_threshold=0.3)
        >>> result = fusion.fuse(facial_result, speech_result, timestamp=1.5)
        >>> print(result.emotions.dominant_emotion)
    """

    def __init__(
        self,
        strategy: Literal["average", "weighted", "max", "confidence"] = "confidence",
        visual_weight: float = 0.6,
        audio_weight: float = 0.4,
        confidence_threshold: float = 0.3,
    ) -> None:
        """Initialize the fusion module.

        Args:
            strategy: Fusion strategy to use.
                - "average": Simple average of all modalities
                - "weighted": Weighted average based on visual/audio weights
                - "max": Take maximum probability for each emotion
                - "confidence": Weight by model confidence scores (default)
            visual_weight: Weight for visual (facial) emotions (0-1).
            audio_weight: Weight for audio (speech) emotions (0-1).
            confidence_threshold: Minimum confidence to include a modality in fusion.
                Results below this threshold are ignored. Set to 0 to disable.
        """
        self.strategy = strategy
        self.visual_weight = visual_weight
        self.audio_weight = audio_weight
        self.confidence_threshold = confidence_threshold

        # Normalize weights
        total = visual_weight + audio_weight
        if total > 0:
            self.visual_weight = visual_weight / total
            self.audio_weight = audio_weight / total

    def fuse(
        self,
        facial_result: FacialEmotionResult | None = None,
        speech_result: SpeechEmotionResult | None = None,
        timestamp: float = 0.0,
    ) -> EmotionResult:
        """Fuse emotion results from multiple modalities.

        Applies confidence thresholding: results with confidence below
        the threshold are treated as if they don't exist.

        Args:
            facial_result: Facial emotion recognition result.
            speech_result: Speech emotion recognition result.
            timestamp: Timestamp for the fused result.

        Returns:
            Fused emotion result.

        Raises:
            ValueError: If no results pass the confidence threshold.
        """
        # Apply confidence thresholding
        facial_valid = (
            facial_result is not None and 
            facial_result.confidence >= self.confidence_threshold
        )
        speech_valid = (
            speech_result is not None and 
            speech_result.confidence >= self.confidence_threshold
        )

        # Use original results only if they pass threshold
        facial_to_use = facial_result if facial_valid else None
        speech_to_use = speech_result if speech_valid else None

        if facial_to_use is None and speech_to_use is None:
            raise ValueError(
                f"No emotion results pass confidence threshold ({self.confidence_threshold}). "
                f"Facial: {facial_result.confidence if facial_result else 'N/A'}, "
                f"Speech: {speech_result.confidence if speech_result else 'N/A'}"
            )

        # Single modality cases (after thresholding)
        if facial_to_use is None and speech_to_use is not None:
            return EmotionResult(
                timestamp=timestamp,
                emotions=speech_to_use.emotions,
                facial_result=facial_result,  # Keep original for reference
                speech_result=speech_to_use,
                fusion_confidence=speech_to_use.confidence,
            )

        if speech_to_use is None and facial_to_use is not None:
            return EmotionResult(
                timestamp=timestamp,
                emotions=facial_to_use.emotions,
                facial_result=facial_to_use,
                speech_result=speech_result,  # Keep original for reference
                fusion_confidence=facial_to_use.confidence,
            )

        # Both modalities available and pass threshold - fuse them
        assert facial_to_use is not None and speech_to_use is not None

        if self.strategy == "average":
            fused_emotions = self._fuse_average(facial_to_use, speech_to_use)
        elif self.strategy == "weighted":
            fused_emotions = self._fuse_weighted(facial_to_use, speech_to_use)
        elif self.strategy == "max":
            fused_emotions = self._fuse_max(facial_to_use, speech_to_use)
        elif self.strategy == "confidence":
            fused_emotions = self._fuse_confidence(facial_to_use, speech_to_use)
        else:
            fused_emotions = self._fuse_weighted(facial_to_use, speech_to_use)

        # Calculate fusion confidence
        fusion_confidence = self._calculate_fusion_confidence(
            facial_to_use, speech_to_use
        )

        return EmotionResult(
            timestamp=timestamp,
            emotions=fused_emotions,
            facial_result=facial_to_use,
            speech_result=speech_to_use,
            fusion_confidence=fusion_confidence,
        )

    def _fuse_average(
        self,
        facial: FacialEmotionResult,
        speech: SpeechEmotionResult,
    ) -> EmotionScores:
        """Simple average fusion."""
        facial_dict = facial.emotions.to_dict()
        speech_dict = speech.emotions.to_dict()

        fused = {
            key: (facial_dict[key] + speech_dict[key]) / 2
            for key in facial_dict
        }

        return EmotionScores.from_dict(fused)

    def _fuse_weighted(
        self,
        facial: FacialEmotionResult,
        speech: SpeechEmotionResult,
    ) -> EmotionScores:
        """Weighted average fusion using predefined weights."""
        facial_dict = facial.emotions.to_dict()
        speech_dict = speech.emotions.to_dict()

        fused = {
            key: (
                facial_dict[key] * self.visual_weight +
                speech_dict[key] * self.audio_weight
            )
            for key in facial_dict
        }

        return EmotionScores.from_dict(fused)

    def _fuse_max(
        self,
        facial: FacialEmotionResult,
        speech: SpeechEmotionResult,
    ) -> EmotionScores:
        """Maximum fusion - take highest probability for each emotion."""
        facial_dict = facial.emotions.to_dict()
        speech_dict = speech.emotions.to_dict()

        fused = {
            key: max(facial_dict[key], speech_dict[key])
            for key in facial_dict
        }

        # Normalize to sum to 1
        total = sum(fused.values())
        if total > 0:
            fused = {k: v / total for k, v in fused.items()}

        return EmotionScores.from_dict(fused)

    def _fuse_confidence(
        self,
        facial: FacialEmotionResult,
        speech: SpeechEmotionResult,
    ) -> EmotionScores:
        """Confidence-weighted fusion - weight by model confidence."""
        facial_dict = facial.emotions.to_dict()
        speech_dict = speech.emotions.to_dict()

        total_conf = facial.confidence + speech.confidence
        if total_conf == 0:
            return self._fuse_average(facial, speech)

        facial_weight = facial.confidence / total_conf
        speech_weight = speech.confidence / total_conf

        fused = {
            key: (
                facial_dict[key] * facial_weight +
                speech_dict[key] * speech_weight
            )
            for key in facial_dict
        }

        return EmotionScores.from_dict(fused)

    def _calculate_fusion_confidence(
        self,
        facial: FacialEmotionResult,
        speech: SpeechEmotionResult,
    ) -> float:
        """Calculate confidence score for the fused result.

        Considers:
        - Individual model confidences
        - Agreement between modalities

        Args:
            facial: Facial emotion result.
            speech: Speech emotion result.

        Returns:
            Fusion confidence score (0-1).
        """
        # Base confidence from individual models
        base_confidence = (
            facial.confidence * self.visual_weight +
            speech.confidence * self.audio_weight
        )

        # Agreement bonus/penalty
        facial_dominant = facial.emotions.dominant_emotion
        speech_dominant = speech.emotions.dominant_emotion

        if facial_dominant == speech_dominant:
            # Modalities agree - boost confidence
            agreement_factor = 1.2
        else:
            # Check if they at least have similar top emotions
            facial_dict = facial.emotions.to_dict()
            speech_dict = speech.emotions.to_dict()

            # Get top 2 emotions from each
            facial_top2 = sorted(facial_dict, key=facial_dict.get, reverse=True)[:2]  # type: ignore
            speech_top2 = sorted(speech_dict, key=speech_dict.get, reverse=True)[:2]  # type: ignore

            if set(facial_top2) & set(speech_top2):
                # Some overlap in top emotions
                agreement_factor = 1.0
            else:
                # Complete disagreement - reduce confidence
                agreement_factor = 0.8

        final_confidence = min(1.0, base_confidence * agreement_factor)
        return final_confidence

    def fuse_multiple(
        self,
        facial_results: list[FacialEmotionResult],
        speech_results: list[SpeechEmotionResult],
        timestamps: list[float] | None = None,
    ) -> list[EmotionResult]:
        """Fuse multiple pairs of emotion results.

        Handles cases where there are different numbers of facial and
        speech results by aligning based on timestamps.

        Args:
            facial_results: List of facial emotion results.
            speech_results: List of speech emotion results.
            timestamps: List of timestamps. If None, uses indices.

        Returns:
            List of fused emotion results.
        """
        if not facial_results and not speech_results:
            return []

        if timestamps is None:
            max_len = max(len(facial_results), len(speech_results))
            timestamps = [float(i) for i in range(max_len)]

        results = []

        # Simple alignment by index for now
        # A more sophisticated approach would align by timestamp
        max_len = max(len(facial_results), len(speech_results))

        for i in range(max_len):
            facial = facial_results[i] if i < len(facial_results) else None
            speech = speech_results[i] if i < len(speech_results) else None
            ts = timestamps[i] if i < len(timestamps) else float(i)

            if facial is not None or speech is not None:
                results.append(self.fuse(facial, speech, ts))

        return results

