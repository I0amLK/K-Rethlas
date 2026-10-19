"""Versioned research-project contracts; fixture protocols stay unchanged."""
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from jsonschema import Draft202012Validator

from .private_fs import strict_json_loads
from .validity import ValidityError
from .validity_protocol import META_BYTES, copied, _strict_integers


VERSION = "project-1"


@lru_cache(maxsize=32)
def validator(kind):
    schema = strict_json_loads(files("rethlas_core").joinpath("schemas", "project.schema.json").read_bytes())
    Draft202012Validator.check_schema(schema)
    if kind not in schema["$defs"]:
        raise ValidityError("unknown research project protocol")
    return Draft202012Validator({"$ref": "#/$defs/" + kind, "$defs": schema["$defs"]})


def protocol(value, kind, maximum=META_BYTES):
    value = copied(value, maximum)
    if not validator(kind).is_valid(value):
        raise ValidityError("invalid " + kind + " research project protocol")
    _strict_integers(value)
    return value
