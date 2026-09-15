"""Order-stable, purpose-scoped child seeds derived from one master seed."""

from __future__ import annotations

import hashlib


def _stable_seed(master_seed: int, purpose: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{purpose}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)
