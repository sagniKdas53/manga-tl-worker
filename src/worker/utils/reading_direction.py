"""One spelling for a series' reading direction.

The UI stores `rtl` / `ltr` / `ttb`, older series and the corpus scripts `rightToLeft` /
`leftToRight`, and the backend sends the stored value lowercased. Grouping compares against the
short forms, so every handler normalises what it receives first (the frontend's
utils/readingDirection.ts accepts the same spellings).
"""

_SPELLINGS = {
    "rtl": "rtl",
    "righttoleft": "rtl",
    "ltr": "ltr",
    "lefttoright": "ltr",
    "ttb": "ttb",
    "toptobottom": "ttb",
    "vertical": "ttb",
    "webtoon": "ttb",
}


def normalize_reading_direction(value) -> str:
    """Return `rtl`, `ltr` or `ttb`. Missing or unknown values stay `rtl`, the old default."""
    key = str(value or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    return _SPELLINGS.get(key, "rtl")
