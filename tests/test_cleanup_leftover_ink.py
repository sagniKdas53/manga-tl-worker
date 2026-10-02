"""Leftover-ink growth: lettering CTD left just outside the mask joins it before the paint."""

import numpy as np

from worker.services.cleanup_reconstruct import CleanupConfig, _grow_by_leftover_ink

BUBBLE = (246, 222, 180)  # BGR of a light-blue bubble
CFG = CleanupConfig()


def _bubble(size=200):
    crop = np.zeros((size, size, 3), dtype=np.uint8)
    crop[:] = BUBBLE
    mask = np.zeros((size, size), dtype=bool)
    mask[80:120, 80:120] = True  # what CTD found, already dilated
    return crop, mask


def _grow(crop, mask, box=(10, 10, 180, 180)):
    x, y, w, h = box
    return _grow_by_leftover_ink(crop, mask, 0, 0, x, y, w, h, CFG)


def test_a_dark_stroke_just_outside_the_mask_joins_it_whole():
    crop, mask = _bubble()
    crop[70:150, 133:139] = (0, 0, 0)  # 14px right of the mask, runs 30px below it
    grown, ink_px = _grow(crop, mask)
    assert ink_px == 80 * 6
    assert grown[70:150, 133:139].all()  # the far end too, not only the part near the mask
    assert grown[mask].all()


def test_a_white_glyph_outline_counts_as_ink_on_a_coloured_bubble():
    crop, mask = _bubble()
    crop[100:108, 121:127] = (255, 255, 255)
    _, ink_px = _grow(crop, mask)
    assert ink_px == 8 * 6


def test_the_bubble_outline_on_the_region_box_edge_is_left_alone():
    crop, mask = _bubble()
    crop[10:190, 10:14] = (0, 0, 0)  # touches the box's left edge
    crop[118:124, 14:90] = (0, 0, 0)  # and reaches the mask
    grown, ink_px = _grow(crop, mask)
    assert ink_px == 0
    assert (grown == mask).all()


def test_ink_out_of_reach_of_the_mask_is_left_alone():
    crop, mask = _bubble()
    crop[30:40, 30:40] = (0, 0, 0)  # 40px from the mask
    _, ink_px = _grow(crop, mask)
    assert ink_px == 0


def test_nothing_grows_when_the_background_around_the_mask_is_not_flat():
    crop, mask = _bubble()
    rng = np.random.default_rng(0)
    crop[:] = rng.integers(0, 256, crop.shape, dtype=np.uint8)  # artwork, not a bubble
    grown, ink_px = _grow(crop, mask)
    assert ink_px == 0
    assert (grown == mask).all()


def test_a_few_stray_pixels_do_not_grow_the_mask():
    crop, mask = _bubble()
    crop[100:104, 122:126] = (0, 0, 0)  # 16px, under the 30px floor
    _, ink_px = _grow(crop, mask)
    assert ink_px == 0
