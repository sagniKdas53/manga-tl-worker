"""Pure, source-preserving region policy selection for ``page-scene/v1``.

Classification is evidence, not permission to replace source pixels.  This module keeps
that boundary explicit so every worker stage can make the same decision without importing
handlers or mutating OCR records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PolicyAction = Literal["preserve", "explain", "replace", "review"]
PolicyKind = Literal["dialogue", "sfx", "caption", "title", "other", "unknown"]

_VALID_ACTIONS = frozenset(("preserve", "explain", "replace", "review"))
_KIND_BY_REGION_TYPE: dict[str, PolicyKind] = {
    "speech": "dialogue",
    "sfx": "sfx",
    "caption": "caption",
    "narration": "other",
    "sign": "other",
}


@dataclass(frozen=True)
class RegionPolicy:
    """One policy decision for one OCR region.

    ``uncertain`` deliberately remains true for every unreviewed classification.  A
    classifier can identify a candidate kind but cannot establish owner identity, glyph
    support, or cleanup authorization.
    """

    kind: PolicyKind
    action: PolicyAction
    reason: str
    user_override: PolicyAction | None
    uncertain: bool

    def callback_fields(self) -> dict[str, str | bool | None]:
        """Serialize the stable policy surface emitted by the layout callback."""
        return {
            "policyKind": self.kind,
            "policyAction": self.action,
            "policyReason": self.reason,
            "policyOverride": self.user_override,
            "policyUncertain": self.uncertain,
        }


def select_region_action(region_type: object, user_override: object = None) -> RegionPolicy:
    """Choose a non-destructive default action for a classified OCR region.

    A valid explicit override is authoritative for that region alone.  Without one,
    including when a classifier says ``sfx``, the evidence is insufficient for automatic
    replacement or implicit suppression, so the action is ``review``.
    """
    normalized_type = region_type if isinstance(region_type, str) else "unknown"
    kind = _KIND_BY_REGION_TYPE.get(normalized_type, "unknown")

    if user_override is not None:
        if user_override in _VALID_ACTIONS:
            action = user_override
            return RegionPolicy(
                kind=kind,
                action=action,
                reason="explicit-user-override",
                user_override=action,
                uncertain=False,
            )
        return RegionPolicy(
            kind=kind,
            action="review",
            reason="invalid-user-override",
            user_override=None,
            uncertain=True,
        )

    if kind == "sfx":
        reason = "classifier-sfx-requires-review"
    elif kind == "unknown":
        reason = "unknown-classification-requires-review"
    else:
        reason = "classifier-evidence-requires-review"
    return RegionPolicy(kind=kind, action="review", reason=reason, user_override=None, uncertain=True)
