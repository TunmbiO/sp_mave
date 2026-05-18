"""Resolve experiment output directories (legacy vs paper track layout)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_results_dir(cfg: dict[str, Any], results_root: Path) -> Path:
    """Return directory for tables/figures/summaries/logs.

    If ``experiment.results_track`` is set, use::

        results_root / "mave" / track / slug /

    Otherwise preserve legacy behaviour::

        results_root / dataset["name"] /

    Optional keys under ``experiment``:
        results_track : str   e.g. "semi_synthetic", "energy", "housing", "friedman"
        results_slug : str    short folder under the track (default: dataset name)
    """
    exp = cfg.get("experiment") or {}
    track = exp.get("results_track")
    if not track:
        return results_root / cfg["dataset"]["name"]
    slug = exp.get("results_slug") or cfg["dataset"]["name"]
    return results_root / "mave" / str(track) / str(slug)
