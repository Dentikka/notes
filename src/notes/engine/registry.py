"""Registries that map IR spec types to instrument and effect implementations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from notes.engine.base import Effect, Instrument

__all__ = ["EFFECTS", "INSTRUMENTS", "build_effect", "build_instrument", "register_effect", "register_instrument"]

INSTRUMENTS: dict[str, type[Instrument]] = {}
EFFECTS: dict[str, type[Effect]] = {}

_I = TypeVar("_I", bound=type[Instrument])
_E = TypeVar("_E", bound=type[Effect])


def register_instrument(name: str) -> Callable[[_I], _I]:
    def decorator(cls: _I) -> _I:
        if name in INSTRUMENTS:
            raise ValueError(f"Instrument type {name!r} is already registered")
        cls.spec_type = name
        INSTRUMENTS[name] = cls
        return cls

    return decorator


def register_effect(name: str) -> Callable[[_E], _E]:
    def decorator(cls: _E) -> _E:
        if name in EFFECTS:
            raise ValueError(f"Effect type {name!r} is already registered")
        cls.spec_type = name
        EFFECTS[name] = cls
        return cls

    return decorator


def _build(spec: Mapping[str, Any], registry: Mapping[str, type], what: str) -> Any:
    params = dict(spec)
    kind = params.pop("type", None)
    cls = registry.get(kind)
    if cls is None:
        raise ValueError(f"Unknown {what} type {kind!r}. Registered: {', '.join(sorted(registry))}")
    return cls(**params)


def build_instrument(spec: Mapping[str, Any]) -> Instrument:
    return _build(spec, INSTRUMENTS, "instrument")


def build_effect(spec: Mapping[str, Any]) -> Effect:
    return _build(spec, EFFECTS, "effect")
