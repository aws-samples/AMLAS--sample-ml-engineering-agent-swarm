"""Bounded tagged values for the fixed official HumanEval parent oracle."""

from __future__ import annotations

import math


def encode(value, depth=0):
    if depth > 16:
        raise ValueError("tagged value depth exceeded")
    kind = type(value)
    if kind in (str, int, bool, type(None)):
        if kind is str and len(value.encode()) > 65536:
            raise ValueError("tagged string bound exceeded")
        if kind is int and value.bit_length() > 4096:
            raise ValueError("tagged integer bound exceeded")
        return {"t": kind.__name__, "v": value}
    if kind is float:
        if not math.isfinite(value):
            raise ValueError("tagged finite float required")
        return {"t": "float", "v": value}
    if kind in (list, tuple):
        if len(value) > 10000:
            raise ValueError("tagged item bound exceeded")
        return {"t": kind.__name__, "v": [encode(item, depth + 1) for item in value]}
    if kind is dict:
        if len(value) > 10000:
            raise ValueError("tagged item bound exceeded")
        return {
            "t": "dict",
            "v": [[encode(key, depth + 1), encode(item, depth + 1)] for key, item in value.items()],
        }
    raise ValueError("unsupported tagged type")


def decode(value, depth=0):
    if depth > 16 or type(value) is not dict or set(value) != {"t", "v"}:
        raise ValueError("tagged response shape differs")
    tag, item = value["t"], value["v"]
    if type(tag) is not str:
        raise ValueError("tagged type name required")
    if tag == "NoneType" and item is None:
        return None
    if tag == "bool" and type(item) is bool:
        return item
    if tag == "int" and type(item) is int and item.bit_length() <= 4096:
        return item
    if tag == "float" and type(item) is float and math.isfinite(item):
        return item
    if tag == "str" and type(item) is str and len(item.encode()) <= 65536:
        return item
    if tag in ("list", "tuple") and type(item) is list and len(item) <= 10000:
        decoded = [decode(element, depth + 1) for element in item]
        return decoded if tag == "list" else tuple(decoded)
    if tag == "dict" and type(item) is list and len(item) <= 10000:
        result = {}
        for pair in item:
            if type(pair) is not list or len(pair) != 2:
                raise ValueError("tagged dictionary pair differs")
            key, child = decode(pair[0], depth + 1), decode(pair[1], depth + 1)
            try:
                if key in result:
                    raise ValueError("tagged duplicate key")
                result[key] = child
            except TypeError:
                raise ValueError("tagged dictionary key must be hashable") from None
        return result
    raise ValueError("tagged value/type differs")
