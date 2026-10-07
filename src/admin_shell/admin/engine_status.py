"""Engine status / settings contracts shared by tts and stt.

EngineStatus mirrors stt-gateway's registry.engine_status(). Settings forms are
described as field specs so the UI renders them without service-specific code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

FIELD_KINDS = frozenset({"select", "text", "number", "checkbox"})


class EngineNotFound(Exception):
    pass


class InvalidSetting(Exception):
    pass


@dataclass(frozen=True)
class EngineStatus:
    name: str
    display_name: str
    description: str
    loaded: bool
    in_flight: int = 0
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "loaded": self.loaded,
            "in_flight": self.in_flight,
            "config": dict(self.config),
        }


class EngineStatusProvider(Protocol):
    async def statuses(self) -> list[EngineStatus]: ...
    async def unload(self, name: str) -> dict[str, Any]:
        """Raise EngineNotFound for unknown names."""
        ...


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    kind: str
    value: Any
    options: list[str] | None = None
    help: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in FIELD_KINDS:
            raise ValueError(
                f"unknown field kind {self.kind!r}; expected one of {sorted(FIELD_KINDS)}"
            )
        if self.kind == "select" and not self.options:
            raise ValueError(f"select field {self.key!r} needs options")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "value": self.value,
            "options": list(self.options) if self.options is not None else None,
            "help": self.help,
        }


@dataclass(frozen=True)
class EngineSettingsForm:
    name: str
    title: str
    fields: list[FieldSpec]
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "note": self.note,
            "fields": [spec.to_dict() for spec in self.fields],
        }


class EngineSettingsProvider(Protocol):
    async def describe(self) -> list[EngineSettingsForm]: ...
    async def apply(self, name: str, values: dict[str, Any]) -> dict[str, Any]:
        """Raise EngineNotFound / InvalidSetting; return the applied values."""
        ...
