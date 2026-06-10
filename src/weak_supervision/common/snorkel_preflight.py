"""Official Snorkel dependency preflight helpers."""

from __future__ import annotations

import importlib.util
from typing import Any


REQUIRED_SNORKEL_IMPORTS = {
    "snorkel.labeling": "LabelingFunction/PandasLFApplier/LFAnalysis",
    "snorkel.labeling.model": "LabelModel",
}


def official_snorkel_preflight() -> dict[str, Any]:
    """Return whether the official Snorkel API surface is importable."""

    modules = {
        module_name: importlib.util.find_spec(module_name) is not None
        for module_name in REQUIRED_SNORKEL_IMPORTS
    }
    available = all(modules.values())
    return {
        "available": available,
        "modules": modules,
        "required_api": REQUIRED_SNORKEL_IMPORTS,
        "mode": "official_snorkel_ready" if available else "local_lf_fallback",
    }
