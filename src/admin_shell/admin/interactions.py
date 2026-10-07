"""Recent activity contract: the reader is bounded and already redacted.

InteractionsFeed pairs a reader with the columns the service declares, so both
the REST route and the SSE provider emit the same {columns, items} shape.
"""

from __future__ import annotations

from typing import Any, Protocol

from .table_spec import ColumnSpec


class InteractionReader(Protocol):
    async def recent(self, limit: int) -> list[dict[str, Any]]: ...


class InteractionsFeed:
    def __init__(
        self, reader: InteractionReader, columns: list[ColumnSpec], *, snapshot_limit: int = 50
    ) -> None:
        self._reader = reader
        self._columns = [column.to_dict() for column in columns]
        self._snapshot_limit = snapshot_limit

    async def recent(self, limit: int) -> dict[str, Any]:
        return {"columns": list(self._columns), "items": await self._reader.recent(limit)}

    async def snapshot(self) -> dict[str, Any]:
        return await self.recent(self._snapshot_limit)
