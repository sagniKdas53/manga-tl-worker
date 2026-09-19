import pytest

from worker.services.region_policy import select_region_action


@pytest.mark.parametrize(
    ("region_type", "expected_kind", "expected_reason"),
    [
        ("speech", "dialogue", "classifier-evidence-requires-review"),
        ("sfx", "sfx", "classifier-sfx-requires-review"),
        ("caption", "caption", "classifier-evidence-requires-review"),
        ("narration", "other", "classifier-evidence-requires-review"),
        ("sign", "other", "classifier-evidence-requires-review"),
        (None, "unknown", "unknown-classification-requires-review"),
    ],
)
def test_unreviewed_classification_is_never_automatic_authorization(region_type, expected_kind, expected_reason):
    policy = select_region_action(region_type)

    assert policy.kind == expected_kind
    assert policy.action == "review"
    assert policy.reason == expected_reason
    assert policy.user_override is None
    assert policy.uncertain is True


@pytest.mark.parametrize("action", ("preserve", "explain", "replace", "review"))
def test_valid_explicit_user_override_wins_for_only_that_region(action):
    policy = select_region_action("sfx", action)

    assert policy.action == action
    assert policy.user_override == action
    assert policy.reason == "explicit-user-override"
    assert policy.uncertain is False


def test_invalid_explicit_override_fails_closed_to_review():
    policy = select_region_action("speech", "erase-everything")

    assert policy.action == "review"
    assert policy.reason == "invalid-user-override"
    assert policy.user_override is None
    assert policy.uncertain is True


def test_callback_surface_retains_action_reason_override_and_uncertainty():
    assert select_region_action("sfx").callback_fields() == {
        "policyKind": "sfx",
        "policyAction": "review",
        "policyReason": "classifier-sfx-requires-review",
        "policyOverride": None,
        "policyUncertain": True,
    }
