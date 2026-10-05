"""Halo growth: a band of outline or glow colour hugging the lettering joins the mask; plates,
labels and plain text are left exactly as they were."""

import hashlib
import os
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from worker.services.cleanup_reconstruct import (
    GENERATOR_SHA256,
    CleanupConfig,
    _dilate,
    _grow_by_halo,
    generator_sha256_for,
    reconstruct_region,
)

CFG = CleanupConfig()
SIZE = 240
BOX = (60, 40, 120, 160)  # x, y, w, h of the region inside the crop


def _art():
    """A smooth diagonal gradient, the kind of artwork text sits on, with a little grain."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    crop = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    crop[..., 0] = 90 + (xx * 60 // SIZE)
    crop[..., 1] = 120 + (yy * 50 // SIZE)
    crop[..., 2] = 150 - (xx * 40 // SIZE)
    rng = np.random.default_rng(0)
    return np.clip(crop.astype(np.int16) + rng.integers(-4, 5, crop.shape), 0, 255).astype(np.uint8)


def _column():
    """Two glyph strokes in a column, and the mask today's steps make of them (glyphs + 5px)."""
    glyphs = np.zeros((SIZE, SIZE), dtype=bool)
    glyphs[60:120, 105:115] = True
    glyphs[130:180, 100:120] = True
    return glyphs, _dilate(glyphs, CFG.mask_dilate_px)


def _grow(crop, mask, config=CFG):
    x, y, w, h = BOX
    return _grow_by_halo(crop, mask, 0, 0, x, y, w, h, config)


def test_a_white_outline_left_outside_the_mask_joins_it():
    crop = _art()
    glyphs, mask = _column()
    outline = _dilate(glyphs, 10) & ~glyphs  # 10px outline: 5px of it survives today's mask
    crop[outline] = (255, 255, 255)
    crop[glyphs] = (20, 20, 20)
    grown, halo_px = _grow(crop, mask)
    assert halo_px > 0
    assert grown[outline].mean() > 0.99  # the white that would seed white blobs is all covered
    assert not grown[~_dilate(glyphs, 13)].any()  # and the art 3px past it is not touched
    assert grown[mask].all()


def test_plain_text_on_artwork_is_left_alone():
    crop = _art()
    glyphs, mask = _column()
    crop[glyphs] = (20, 20, 20)
    grown, halo_px = _grow(crop, mask)
    assert halo_px == 0
    assert (grown == mask).all()


def test_an_outline_already_inside_the_mask_changes_nothing():
    crop = _art()
    glyphs, mask = _column()
    crop[_dilate(glyphs, 3)] = (255, 255, 255)  # thin outline, covered by the 5px dilation
    crop[glyphs] = (20, 20, 20)
    grown, halo_px = _grow(crop, mask)
    assert halo_px == 0
    assert (grown == mask).all()


def test_a_plate_that_ends_unevenly_round_the_text_is_left_alone():
    """Dark text on a pink plate inside a white balloon: the plate edge is 1px from one glyph
    and 9px from another, so it is a shape of its own, not an outline."""
    crop = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
    glyphs, mask = _column()
    crop[54:186, 96:136] = (200, 190, 245)
    crop[glyphs] = (40, 40, 40)
    grown, halo_px = _grow(crop, mask)
    assert halo_px == 0
    assert (grown == mask).all()


def test_a_plate_that_runs_on_past_the_band_is_left_alone():
    crop = _art()
    glyphs, mask = _column()
    crop[40:200, 60:180] = (255, 255, 255)  # a label the size of the region box
    crop[glyphs] = (20, 20, 20)
    grown, halo_px = _grow(crop, mask)
    assert halo_px == 0
    assert (grown == mask).all()


def test_text_in_a_plain_balloon_skips_the_background_fill():
    """White balloon, dark text: both rings are the balloon's white, so the step returns before
    the fill that is most of its cost."""
    crop = np.full((SIZE, SIZE, 3), 250, dtype=np.uint8)
    glyphs, mask = _column()
    crop[glyphs] = (20, 20, 20)
    with patch("worker.services.cleanup_reconstruct._local_background") as fill:
        grown, halo_px = _grow(crop, mask)
    fill.assert_not_called()
    assert halo_px == 0
    assert (grown == mask).all()


def test_an_outline_on_a_flat_colour_still_runs_the_fill():
    """A white outline on a flat blue ground: the outer ring is flat, but the inner one is not
    the same colour, so the check goes on and the outline joins the mask."""
    crop = np.full((SIZE, SIZE, 3), (200, 120, 40), dtype=np.uint8)
    glyphs, mask = _column()
    outline = _dilate(glyphs, 10) & ~glyphs
    crop[outline] = (255, 255, 255)
    crop[glyphs] = (20, 20, 20)
    grown, halo_px = _grow(crop, mask)
    assert halo_px > 0
    assert grown[outline].mean() > 0.99


@pytest.mark.parametrize("label", [(187, 236, 225), (130, 234, 247), (176, 183, 208)])
def test_a_tinted_label_cut_to_fit_the_text_is_left_alone(label):
    """A pale green, yellow or beige label hugging a column ends as evenly as an outline does; only
    its colour tells it apart (the three tints are from corpus pages it would have erased)."""
    crop = _art()
    glyphs, mask = _column()
    crop[_dilate(glyphs, 10) & ~glyphs] = label
    crop[glyphs] = (20, 20, 20)
    grown, halo_px = _grow(crop, mask)
    assert halo_px == 0
    assert (grown == mask).all()


@pytest.mark.parametrize("enabled", [True, False])
def test_reconstruct_region_paints_the_grown_mask_only_when_switched_on(enabled):
    crop = _art()
    glyphs, _ = _column()
    outline = _dilate(glyphs, 10) & ~glyphs
    crop[outline] = (255, 255, 255)
    crop[glyphs] = (20, 20, 20)
    config = CleanupConfig(halo_grow=enabled, ink_grow=False)
    painted = {}

    def fake_aot(crop_bgr, mask_bool, _config):
        painted["mask"] = mask_bool.copy()
        return crop_bgr.copy()

    x, y, w, h = BOX
    with (
        patch("worker.services.cleanup_reconstruct.segment_crop", return_value=glyphs.astype(np.float32)),
        patch("worker.services.cleanup_reconstruct._reconstruct_aot", side_effect=fake_aot),
    ):
        result = reconstruct_region(crop, x, y, w, h, config=config, mode="aot")
    assert result is not None
    assert bool(painted["mask"][outline].mean() > 0.99) == enabled
    assert any("halo added to the mask" in d for d in result.diagnostics) == enabled


def test_auto_mode_paints_a_grown_halo_with_aot_even_when_the_mask_reads_flat():
    crop = _art()
    glyphs, _ = _column()
    outline = _dilate(glyphs, 10) & ~glyphs
    crop[outline] = (255, 255, 255)
    crop[glyphs] = (20, 20, 20)
    x, y, w, h = BOX
    with (
        patch("worker.services.cleanup_reconstruct.segment_crop", return_value=glyphs.astype(np.float32)),
        patch("worker.services.cleanup_reconstruct.pixel_spread", return_value=0.0),  # "flat" interior
        patch("worker.services.cleanup_reconstruct._reconstruct_aot", side_effect=lambda c, m, _cfg: c.copy()) as aot,
        patch("worker.services.cleanup_reconstruct._reconstruct_telea", side_effect=lambda c, m: c.copy()) as telea,
    ):
        grown = reconstruct_region(crop, x, y, w, h, config=CleanupConfig(ink_grow=False), mode="auto")
        plain = reconstruct_region(crop, x, y, w, h, config=CleanupConfig(ink_grow=False, halo_grow=False), mode="auto")
    assert grown is not None and plain is not None
    assert aot.call_count == 1 and telea.call_count == 1
    assert any("reconstruction method: aot" in d for d in grown.diagnostics)
    assert any("reconstruction method: telea" in d for d in plain.diagnostics)


@pytest.mark.parametrize(("value", "expected"), [("false", "False"), ("0", "False"), ("true", "True"), (None, "True")])
def test_the_cleanup_halo_grow_env_var_sets_the_default(value, expected):
    """CLEANUP_HALO_GROW=false switches halo growth off for every page; unset leaves it on. Read at
    import, so each case gets a fresh interpreter."""
    env = {k: v for k, v in os.environ.items() if k != "CLEANUP_HALO_GROW"}
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    if value is not None:
        env["CLEANUP_HALO_GROW"] = value
    code = "from worker.services.cleanup_reconstruct import CleanupConfig; print(CleanupConfig().halo_grow)"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == expected


def test_with_halo_growth_off_the_patch_records_the_v3_recipe():
    """Switched off, cleanup makes exactly v3's masks and patches, so it must not label them v4."""
    v3 = hashlib.sha256(b"ctd-seg+telea-aotgan-cleanup/v3-leftover-ink").hexdigest()
    assert generator_sha256_for("auto", halo_grow=False) == v3
    assert generator_sha256_for("auto") == GENERATOR_SHA256 != v3
    assert generator_sha256_for("aot", halo_grow=False) != generator_sha256_for("aot")
