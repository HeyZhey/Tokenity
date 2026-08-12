# SPDX-License-Identifier: Apache-2.0
"""Exact depth-one rollback for Qwen hybrid caches.

Adapted from oMLX ``mlx_lm_mtp/cache_rollback.py`` and mlx-lm PR #990.
Tokenity keeps only the Qwen singleton depth-one path: ArraysCache snapshots
linear state, while rotating KV caches keep a one-update undo record for the
two-token verify window.
"""

from __future__ import annotations

import threading
from typing import Any

from .runtime import PatchTransaction


_UNDO_ARMED = threading.local()


def set_undo_armed(flag: bool) -> None:
    _UNDO_ARMED.value = bool(flag)


def _is_undo_armed() -> bool:
    return bool(getattr(_UNDO_ARMED, "value", False))


def install(transaction: PatchTransaction) -> None:
    from mlx_lm.models.cache import (  # type: ignore
        ArraysCache,
        BatchRotatingKVCache,
        RotatingKVCache,
    )

    if not hasattr(ArraysCache, "rollback_state"):
        transaction.set(ArraysCache, "rollback_state", None)
    transaction.set(ArraysCache, "_tokenity_native_mtp_cache_patch", True)

    _wrap_rotating(
        transaction,
        RotatingKVCache,
        ("keys", "values", "offset", "_idx"),
    )
    _wrap_rotating(
        transaction,
        BatchRotatingKVCache,
        ("keys", "values", "offset", "_offset", "_idx", "rotated", "left_padding"),
    )


def _wrap_rotating(
    transaction: PatchTransaction,
    cls: type,
    fields: tuple[str, ...],
) -> None:
    if getattr(cls, "_tokenity_native_mtp_undo_patch", False):
        return

    import mlx.core as mx  # type: ignore

    original_update = cls.update_and_fetch
    original_is_trimmable = cls.is_trimmable
    original_trim = cls.trim

    def update_and_fetch(self: Any, keys: Any, values: Any) -> Any:
        if keys.shape[2] == 2 and _is_undo_armed():
            snapshot: dict[str, Any] = {}
            for field in fields:
                value = getattr(self, field)
                if isinstance(value, mx.array):
                    value = value + 0
                snapshot[field] = value
            self._tokenity_mtp_undo = (snapshot, keys, values)
        else:
            self._tokenity_mtp_undo = None
        return original_update(self, keys, values)

    def is_trimmable(self: Any) -> bool:
        return bool(
            original_is_trimmable(self)
            or getattr(self, "_tokenity_mtp_undo", None) is not None
        )

    def trim(self: Any, count: int) -> int:
        if original_is_trimmable(self):
            self._tokenity_mtp_undo = None
            return original_trim(self, count)
        undo = getattr(self, "_tokenity_mtp_undo", None)
        self._tokenity_mtp_undo = None
        if undo is None or count != 1:
            return 0
        snapshot, keys, values = undo
        for field, value in snapshot.items():
            setattr(self, field, value)
        # Keep the confirmed first token and discard only the draft token.
        original_update(self, keys[..., :1, :], values[..., :1, :])
        self._tokenity_mtp_undo = None
        return 1

    transaction.set(cls, "update_and_fetch", update_and_fetch)
    transaction.set(cls, "is_trimmable", is_trimmable)
    transaction.set(cls, "trim", trim)
    transaction.set(cls, "_tokenity_mtp_undo", None)
    transaction.set(cls, "_tokenity_native_mtp_undo_patch", True)


def restore_after_rejection(prompt_cache: list[Any]) -> bool:
    """Check every layer first, then restore exactly the rejected draft."""

    for cache in prompt_cache:
        if getattr(cache, "rollback_state", None) is not None:
            continue
        if hasattr(cache, "is_trimmable") and cache.is_trimmable():
            continue
        return False

    for cache in prompt_cache:
        rollback = getattr(cache, "rollback_state", None)
        if rollback is not None:
            cache[0], cache[1] = rollback
            cache.rollback_state = None
        else:
            if cache.trim(1) != 1:
                return False
    return True


def clear_rollback(prompt_cache: list[Any]) -> None:
    for cache in prompt_cache:
        if hasattr(cache, "rollback_state"):
            cache.rollback_state = None
        if hasattr(cache, "_tokenity_mtp_undo"):
            cache._tokenity_mtp_undo = None
        for child in getattr(cache, "caches", ()):
            if hasattr(child, "_tokenity_mtp_undo"):
                child._tokenity_mtp_undo = None

