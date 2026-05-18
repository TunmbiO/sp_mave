"""Paper-facing method names (internal estimators unchanged)."""

from __future__ import annotations

import re
from typing import Any


def paper_method_name(internal_name: str) -> str:
    """Map estimator ``method_name`` strings to paper labels.

    - ClassicalMAVE -> ClassicalMAVE
    - ClassicalMAVE_k* -> ClassicalMAVE_kNN
    - SPMAVE -> RFWeightedMAVE
    """
    if internal_name == "SPMAVE":
        return "RFWeightedMAVE"
    if internal_name == "ClassicalMAVE":
        return "ClassicalMAVE"
    if re.match(r"^ClassicalMAVE_k\d+$", internal_name):
        return "ClassicalMAVE_kNN"
    return internal_name


def aggregation_group_key(row: dict[str, Any]) -> str:
    """Key for multi-seed aggregation / summary tables (one row per paper method).

    Uses ``method_paper`` when present (from run_experiment), else derives from
    ``method``.  This merges ``ClassicalMAVE_k20``, ``ClassicalMAVE_k25``, …
    into a single **ClassicalMAVE_kNN** bucket, comparable to **RFWeightedMAVE**
    always being one bucket regardless of tuned *k*.
    """
    mp = row.get("method_paper")
    if mp is not None and str(mp).strip() != "":
        return str(mp)
    return paper_method_name(str(row.get("method", "")))
