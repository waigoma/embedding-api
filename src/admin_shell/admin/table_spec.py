"""Column vocabulary shared by the server contract and the UI table renderer.

The server declares *what* each column holds; the UI decides *how* to draw a
kind. Adding a kind means adding it here and in static/admin/core/table.js.
"""

from __future__ import annotations

from dataclasses import dataclass

CELL_KINDS = frozenset(
    {
        "text",
        "mono",
        "status",
        "bytes",
        "duration",
        "time",
        "path-list",
        "actions",
    }
)
MODEL_STATUSES = frozenset({"installed", "queued", "downloading", "completed", "failed"})


@dataclass(frozen=True)
class ColumnSpec:
    key: str
    label: str
    kind: str

    def __post_init__(self) -> None:
        if not self.key or self.key != self.key.strip():
            raise ValueError("column key must be a nonempty trimmed string")
        if self.kind not in CELL_KINDS:
            raise ValueError(
                f"unknown cell kind {self.kind!r}; expected one of {sorted(CELL_KINDS)}"
            )

    def to_dict(self) -> dict[str, str]:
        return {"key": self.key, "label": self.label, "kind": self.kind}
