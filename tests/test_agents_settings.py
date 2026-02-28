from backend.services.agents import build_model_settings


def test_build_model_settings_applies_reasoning_effort() -> None:
    """Reasoning effort is forwarded into model settings when configured."""
    settings = build_model_settings(
        temperature=0.0,
        max_output_tokens=None,
        reasoning_effort="none",
    )

    assert settings.reasoning is not None
    assert settings.reasoning.effort == "none"


def test_build_model_settings_without_reasoning_effort() -> None:
    """Reasoning settings remain unset when no effort is provided."""
    settings = build_model_settings(
        temperature=0.0,
        max_output_tokens=None,
    )

    assert settings.reasoning is None
