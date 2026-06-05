"""Manuscript display names for models.

Public name: Hybrid-STEMSeg. Legacy logs may say Hybrid-NoGAN; checkpoint dir/key
remains ``hybrid-nogan`` for weight compatibility.
"""

from __future__ import annotations

# Primary method name in figures and tables.
HYBRID_STEMSEG = "Hybrid-STEMSeg"

# Legacy label used in older logs/CSVs; map on read/plot.
LEGACY_HYBRID_NOGAN = "Hybrid-NoGAN"


def hybrid_display_name(variant: str = "") -> str:
    if variant:
        return f"{HYBRID_STEMSEG} ({variant})"
    return HYBRID_STEMSEG


def normalize_display_name(name: str) -> str:
    """Map stored model column / legend strings to current manuscript naming."""
    if name == LEGACY_HYBRID_NOGAN or name == "hybrid-nogan":
        return HYBRID_STEMSEG
    return name.replace(LEGACY_HYBRID_NOGAN, HYBRID_STEMSEG)
