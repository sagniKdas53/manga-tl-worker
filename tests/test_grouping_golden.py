"""Grouping on 91 real pages' piece positions must not change without someone meaning it to.

The fixture holds coordinates only: no page images and no OCR text, so it can live in a public repo.
It comes from the test stack's OCR captures. When a change to grouping is meant, regenerate it with
scripts/quality/make_grouping_fixture.py in the parent repo, read what moved, and add a row to the
layout ledger (docs/quality-runs/layout-regression/README.md).
"""

import json
from pathlib import Path

import pytest

from tests.grouping_replay import group_page

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "grouping_golden.json").read_text())


def test_the_fixture_holds_coordinates_and_no_text():
    for page in FIXTURE["pages"]:
        for piece in page["pieces"]:
            assert set(piece) == {"x", "y", "width", "height", "sourceQuad", "fragmentId"}


def test_the_fixture_covers_both_paths():
    with_balloons = [page for page in FIXTURE["pages"] if page["balloons"]]
    without_balloons = [page for page in FIXTURE["pages"] if not page["balloons"]]
    assert len(FIXTURE["pages"]) >= 90
    assert len(with_balloons) >= 50
    assert without_balloons


@pytest.mark.parametrize("page", FIXTURE["pages"], ids=lambda page: page["page"])
def test_grouping_is_unchanged(page):
    assert group_page(page) == page["groups"], (
        "Grouping changed on this page. If that's meant, regenerate the fixture with "
        "scripts/quality/make_grouping_fixture.py and add a row to the layout ledger."
    )
