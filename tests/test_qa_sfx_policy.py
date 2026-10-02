"""QA enforces "sound effects are never typeset" (user policy, 2026-08-13).

QA used to be asked to reject "a sound effect or gibberish that shouldn't be translated", without
being told which regions the layout classifier took for SFX, while the translator was told to give
every SFX an English sound word. It passed 32 of 36 SFX on one chapter, and each was drawn on a flat
plate. The prompt now carries each region's type and states the policy outright.
"""

from worker.handlers.qa import REJECT_SFX_RULE, _region_type, _vlm_qa_prompt


def test_the_vision_prompt_names_each_regions_type_and_states_the_policy():
    target = {
        "id": "r1",
        "text": "シュル",
        "translatedText": "SLUR",
        "regionType": "sfx",
        "bboxX": 1,
        "bboxY": 2,
        "bboxW": 3,
        "bboxH": 4,
    }
    prompt = _vlm_qa_prompt([target], [], {"r1": 1})
    assert '"regionType": "sfx"' in prompt
    assert REJECT_SFX_RULE in prompt
    assert "never typeset" in REJECT_SFX_RULE
    assert "even when its English sound word is accurate" in REJECT_SFX_RULE
    # 2026-10-02: an SFX without a patch is not drawn before QA keeps it, so QA must not fail it
    # (and buy a paid retry) for English it cannot see on the page.
    assert "not drawn on the page yet" in REJECT_SFX_RULE
    assert "never fail one for its English missing from the page" in REJECT_SFX_RULE


def test_a_region_without_a_type_reads_as_speech():
    assert _region_type({}) == "speech"
    assert _region_type({"region_type": "sfx"}) == "sfx"
