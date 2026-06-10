"""Tests for weak signal JSON Schema graph labels."""

import json
from pathlib import Path

from src.weak_supervision.common.graph_schema import ENTITY_LABELS


def test_weak_signal_schema_entity_type_enum_matches_graph_labels() -> None:
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "weak_signal_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    enum_values = schema["properties"]["entity_type"]["enum"]
    assert enum_values == ENTITY_LABELS
