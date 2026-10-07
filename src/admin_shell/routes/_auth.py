"""Shared helper: turn an optional auth dependency into router dependencies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends

AuthDependency = Callable[..., Any] | None


def router_dependencies(auth: AuthDependency) -> list[Any]:
    # None is an explicit decision by the composition root (e.g. a LAN-only
    # service that already exposes /v1 unauthenticated), never a default.
    return [Depends(auth)] if auth is not None else []
