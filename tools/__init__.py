"""Inspection / manipulation tools for IF/ELSE-annotated Stim circuits.

These utilities operate on the text dialect produced by
``ConditionalCircuit.to_stim_text`` (``IF(rec[k]) { ... } ELSE { ... }``
blocks) and are intended for ad-hoc inspection, debugging, and feeding
into downstream tooling that expects vanilla Stim circuits.
"""

from tools.measurements import (
    MeasurementInfo,
    build_measurement_map,
    locate_record,
    observable_records,
)
from tools.resolve import resolve_if_else, resolve_if_else_by_order

__all__ = [
    "MeasurementInfo",
    "build_measurement_map",
    "locate_record",
    "observable_records",
    "resolve_if_else",
    "resolve_if_else_by_order",
]
