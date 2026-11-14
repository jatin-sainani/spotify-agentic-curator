"""Test that all modules can be imported successfully."""

import pytest


def test_core_imports():
    """Test core module imports."""
    try:
        from core.config import Settings, settings
        from core.logging import logger, setup_logging  # noqa: F401 (import check)
        from core.models import Draft, FeedbackEvent, Intent, PlanStep, Track  # noqa: F401 (import check)

        # Verify settings is initialized
        assert settings is not None
        assert isinstance(settings, Settings)

        # Verify logger is initialized
        assert logger is not None

        # Verify models can be instantiated
        track = Track(
            id="test_id",
            name="Test Track",
            artist="Test Artist",
            album="Test Album",
            duration_ms=300000
        )
        assert track.id == "test_id"

    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")


def test_app_imports():
    """Test app module imports."""
    try:
        import app  # noqa: F401 (import check)
        from app import commands
        from app.__main__ import app as main_app

        # Verify main app is typer app
        assert main_app is not None

        # Verify command apps exist
        assert hasattr(commands, 'connect_app')
        assert hasattr(commands, 'tools_app')
        assert hasattr(commands, 'curate_app')
        assert hasattr(commands, 'draft_app')

    except ImportError as e:
        pytest.fail(f"Failed to import app modules: {e}")


def test_all_models():
    """Test all Pydantic models can be created."""
    from core.models import Draft, FeedbackEvent, FeedbackType, Intent, IntentType, PlanAction, PlanStep, Track

    # Test Track
    track = Track(
        id="track_123",
        name="Test Song",
        artist="Test Artist",
        album="Test Album",
        duration_ms=240000,
        popularity=85
    )
    assert track.id == "track_123"

    # Test FeedbackEvent
    feedback = FeedbackEvent(
        track_id="track_123",
        feedback_type=FeedbackType.LIKE
    )
    assert feedback.feedback_type == FeedbackType.LIKE

    # Test Intent
    intent = Intent(
        type=IntentType.DISCOVER_NEW_MUSIC,
        description="Find new music",
        confidence=0.9
    )
    assert intent.type == IntentType.DISCOVER_NEW_MUSIC

    # Test Plan (PlanStep)
    plan_step = PlanStep(
        action=PlanAction.SEARCH_TRACKS,
        description="Search for tracks",
        order=1
    )
    assert plan_step.action == PlanAction.SEARCH_TRACKS

    # Test Draft
    draft = Draft(
        id="draft_123",
        name="Test Draft",
        intent=intent,
        tracks=[track]
    )
    assert draft.id == "draft_123"
    assert len(draft.tracks) == 1
