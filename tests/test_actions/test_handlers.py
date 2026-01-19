"""Tests for action handlers."""

import pytest

from emotion_detector.actions.base import BaseActionHandler
from emotion_detector.actions.logging_handler import LoggingActionHandler, MockActionHandler
from emotion_detector.core.types import (
    ActionCommand,
    EmotionLabel,
    EmotionResult,
    EmotionScores,
)


class TestBaseActionHandler:
    """Tests for BaseActionHandler abstract class."""

    def test_generate_default_action_happy(self):
        """Test default action for happy emotion."""
        handler = LoggingActionHandler(verbose=False)
        scores = EmotionScores(happy=0.9)
        emotion = EmotionResult(timestamp=0.0, emotions=scores, fusion_confidence=0.9)

        action = handler._generate_default_action(emotion)

        assert action.action_type == "acknowledge"
        assert action.parameters.get("gesture") == "nod"

    def test_generate_default_action_sad(self):
        """Test default action for sad emotion."""
        handler = LoggingActionHandler(verbose=False)
        scores = EmotionScores(sad=0.9)
        emotion = EmotionResult(timestamp=0.0, emotions=scores, fusion_confidence=0.9)

        action = handler._generate_default_action(emotion)

        assert action.action_type == "comfort"

    def test_generate_default_action_angry(self):
        """Test default action for angry emotion."""
        handler = LoggingActionHandler(verbose=False)
        scores = EmotionScores(angry=0.9)
        emotion = EmotionResult(timestamp=0.0, emotions=scores, fusion_confidence=0.9)

        action = handler._generate_default_action(emotion)

        assert action.action_type == "de_escalate"

    def test_generate_default_action_neutral(self):
        """Test default action for neutral emotion."""
        handler = LoggingActionHandler(verbose=False)
        scores = EmotionScores(neutral=0.9)
        emotion = EmotionResult(timestamp=0.0, emotions=scores, fusion_confidence=0.9)

        action = handler._generate_default_action(emotion)

        assert action.action_type == "idle"

    def test_get_supported_actions(self):
        """Test getting list of supported actions."""
        handler = LoggingActionHandler(verbose=False)
        actions = handler.get_supported_actions()

        assert "idle" in actions
        assert "acknowledge" in actions
        assert "comfort" in actions

    def test_validate_action_valid(self):
        """Test validating a valid action."""
        handler = LoggingActionHandler(verbose=False)
        action = ActionCommand(action_type="acknowledge")

        is_valid, error = handler.validate_action(action)

        assert is_valid is True
        assert error == ""

    def test_validate_action_invalid(self):
        """Test validating an invalid action."""
        handler = LoggingActionHandler(verbose=False)
        action = ActionCommand(action_type="unsupported_action_xyz")

        is_valid, error = handler.validate_action(action)

        assert is_valid is False
        assert "Unsupported action type" in error


class TestLoggingActionHandler:
    """Tests for LoggingActionHandler."""

    def test_connect_disconnect(self):
        """Test connect and disconnect."""
        handler = LoggingActionHandler(verbose=False)

        assert not handler.is_connected
        result = handler.connect()
        assert result is True
        assert handler.is_connected

        handler.disconnect()
        assert not handler.is_connected

    def test_execute_records_action(self):
        """Test that execute records action in history."""
        handler = LoggingActionHandler(verbose=False)
        handler.connect()

        action = ActionCommand(
            action_type="greeting",
            parameters={"message": "hello"},
            confidence=0.9,
        )
        result = handler.execute(action)

        assert result is True
        history = handler.get_action_history()
        assert len(history) == 1
        assert history[0]["action_type"] == "greeting"
        assert history[0]["parameters"]["message"] == "hello"

    def test_execute_multiple_actions(self):
        """Test executing multiple actions."""
        handler = LoggingActionHandler(verbose=False)
        handler.connect()

        handler.execute(ActionCommand(action_type="action1"))
        handler.execute(ActionCommand(action_type="action2"))
        handler.execute(ActionCommand(action_type="action3"))

        history = handler.get_action_history()
        assert len(history) == 3
        assert history[0]["index"] == 1
        assert history[2]["index"] == 3

    def test_get_last_action(self):
        """Test getting the last executed action."""
        handler = LoggingActionHandler(verbose=False)
        handler.connect()

        handler.execute(ActionCommand(action_type="first"))
        handler.execute(ActionCommand(action_type="last"))

        last = handler.get_last_action()
        assert last is not None
        assert last["action_type"] == "last"

    def test_get_last_action_empty(self):
        """Test getting last action when none executed."""
        handler = LoggingActionHandler(verbose=False)
        assert handler.get_last_action() is None

    def test_clear_history(self):
        """Test clearing action history."""
        handler = LoggingActionHandler(verbose=False)
        handler.connect()

        handler.execute(ActionCommand(action_type="test"))
        assert len(handler.get_action_history()) == 1

        handler.clear_history()
        assert len(handler.get_action_history()) == 0

    def test_get_statistics(self):
        """Test getting action statistics."""
        handler = LoggingActionHandler(verbose=False)
        handler.connect()

        handler.execute(ActionCommand(action_type="greeting", confidence=0.9))
        handler.execute(ActionCommand(action_type="greeting", confidence=0.8))
        handler.execute(ActionCommand(action_type="idle", confidence=1.0))

        stats = handler.get_statistics()

        assert stats["total_actions"] == 3
        assert stats["action_types"]["greeting"] == 2
        assert stats["action_types"]["idle"] == 1
        assert stats["avg_confidence"] == pytest.approx(0.9, abs=0.01)

    def test_get_statistics_empty(self):
        """Test statistics when no actions executed."""
        handler = LoggingActionHandler(verbose=False)
        stats = handler.get_statistics()

        assert stats["total_actions"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_callback(self):
        """Test custom callback on execute."""
        received_actions = []

        def callback(action: ActionCommand) -> None:
            received_actions.append(action.action_type)

        handler = LoggingActionHandler(verbose=False, callback=callback)
        handler.connect()

        handler.execute(ActionCommand(action_type="test1"))
        handler.execute(ActionCommand(action_type="test2"))

        assert received_actions == ["test1", "test2"]

    def test_context_manager(self):
        """Test using handler as context manager."""
        with LoggingActionHandler(verbose=False) as handler:
            assert handler.is_connected
            handler.execute(ActionCommand(action_type="test"))

        assert not handler.is_connected

    def test_execute_for_emotion(self):
        """Test executing action based on emotion."""
        handler = LoggingActionHandler(verbose=False)
        handler.connect()

        scores = EmotionScores(happy=0.9)
        emotion = EmotionResult(timestamp=0.0, emotions=scores, fusion_confidence=0.9)

        result = handler.execute_for_emotion(emotion)

        assert result is True
        last = handler.get_last_action()
        assert last["action_type"] == "acknowledge"


class TestMockActionHandler:
    """Tests for MockActionHandler."""

    def test_expect_and_verify_success(self):
        """Test expecting and verifying actions successfully."""
        handler = MockActionHandler()
        handler.connect()

        handler.expect_action("greeting")
        handler.expect_action("idle")

        handler.execute(ActionCommand(action_type="greeting"))
        handler.execute(ActionCommand(action_type="idle"))

        success, message = handler.verify_expectations()
        assert success is True
        assert message == "All expectations met"

    def test_expect_and_verify_failure_wrong_action(self):
        """Test verification fails with wrong action."""
        handler = MockActionHandler()
        handler.connect()

        handler.expect_action("greeting")
        handler.execute(ActionCommand(action_type="idle"))

        success, message = handler.verify_expectations()
        assert success is False
        assert "expected 'greeting'" in message

    def test_expect_and_verify_failure_wrong_count(self):
        """Test verification fails with wrong number of actions."""
        handler = MockActionHandler()
        handler.connect()

        handler.expect_action("greeting")
        handler.expect_action("idle")

        handler.execute(ActionCommand(action_type="greeting"))

        success, message = handler.verify_expectations()
        assert success is False
        assert "Expected 2 actions" in message

    def test_expect_actions_list(self):
        """Test expecting multiple actions at once."""
        handler = MockActionHandler()
        handler.connect()

        handler.expect_actions(["a", "b", "c"])

        handler.execute(ActionCommand(action_type="a"))
        handler.execute(ActionCommand(action_type="b"))
        handler.execute(ActionCommand(action_type="c"))

        success, _ = handler.verify_expectations()
        assert success is True

    def test_assert_action_executed(self):
        """Test asserting specific action was executed."""
        handler = MockActionHandler()
        handler.connect()

        handler.execute(ActionCommand(action_type="greeting"))
        handler.execute(ActionCommand(action_type="idle"))

        handler.assert_action_executed("greeting")  # Should not raise
        handler.assert_action_executed("idle")  # Should not raise

    def test_assert_action_executed_raises(self):
        """Test assertion raises when action not executed."""
        handler = MockActionHandler()
        handler.connect()

        handler.execute(ActionCommand(action_type="greeting"))

        with pytest.raises(AssertionError, match="Action 'idle' was not executed"):
            handler.assert_action_executed("idle")

    def test_assert_no_actions(self):
        """Test asserting no actions were executed."""
        handler = MockActionHandler()
        handler.connect()

        handler.assert_no_actions()  # Should not raise

    def test_assert_no_actions_raises(self):
        """Test assertion raises when actions were executed."""
        handler = MockActionHandler()
        handler.connect()

        handler.execute(ActionCommand(action_type="greeting"))

        with pytest.raises(AssertionError, match="Expected no actions"):
            handler.assert_no_actions()

    def test_reset_expectations(self):
        """Test resetting expectations and history."""
        handler = MockActionHandler()
        handler.connect()

        handler.expect_action("test")
        handler.execute(ActionCommand(action_type="test"))

        handler.reset_expectations()

        assert len(handler.get_action_history()) == 0
        success, _ = handler.verify_expectations()
        assert success is True  # No expectations, no actions

