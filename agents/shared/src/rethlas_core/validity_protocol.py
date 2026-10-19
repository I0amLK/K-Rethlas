"""Strict, bounded controller protocol data; no authority or mathematical review."""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator

from .private_fs import canonical_json, strict_json_loads
from .validity import ValidityError, ValidityCapacity


STATE_BYTES = 4194304
TRANSITION_BYTES = 8388608
REQUEST_BYTES = 524288
DTO_BYTES = 8388608
USE_BYTES = DTO_BYTES + 524288
ROOT_BYTES = 268435456
CONTROL_BYTES = 134217728
STORE_BYTES = 67108864
ROOT_FILES = 8192
META_BYTES = 65536


@lru_cache(maxsize=32)
def validator(kind: str) -> Draft202012Validator:
    schema = strict_json_loads(files("rethlas_core").joinpath("schemas", "validity.schema.json").read_bytes())
    Draft202012Validator.check_schema(schema)
    if kind not in schema["$defs"]:
        raise ValidityError("unknown validity protocol")
    return Draft202012Validator({"$ref": "#/$defs/" + kind, "$defs": schema["$defs"]})


def copied(value: Any, maximum: int = META_BYTES) -> Any:
    raw = canonical_json(value)
    if len(raw) > maximum:
        raise ValidityCapacity("validity object byte capacity reached")
    return strict_json_loads(raw)


def _strict_integers(value: Any) -> None:
    # This protocol has no floating-point fields. JSON Schema alone accepts 1.0
    # as an integer; capability timestamps are the sole trusted-number exception.
    if type(value) is float:
        raise ValidityError("validity counters must be strict integers")
    if type(value) is list:
        for item in value:
            _strict_integers(item)
    elif type(value) is dict:
        for key, item in value.items():
            if key != "authorization":
                _strict_integers(item)


def protocol(value: Any, kind: str, maximum: int = META_BYTES) -> Any:
    value = copied(value, maximum)
    if not validator(kind).is_valid(value):
        raise ValidityError("invalid " + kind + " validity protocol")
    _strict_integers(value)
    return value
