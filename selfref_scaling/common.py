"""Canonical JSON, hashing and seed helpers shared by every stage.

Two JSON encodings are deliberate. ``canonical`` (ASCII) is used for plan
files, receipts and hashes. ``canonical_text`` (UTF-8, unescaped) is the exact
encoding the inherited structured judge instrument uses for its query/response
payload; keep it byte-identical to CONSCIOUS
``experiments/automated_rubric_audit/common.py::canonical``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = "selfref-scaling-qwen397b-20261003"
POD_PREFIX = "scaling-qwen-20261003-"


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def canonical_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def text_digest(value) -> str:
    """Digest used by inherited judge receipts (UTF-8 canonical encoding)."""
    return hashlib.sha256(canonical_text(value).encode()).hexdigest()


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def strict_json(raw):
    """Reject duplicate keys and non-finite numbers."""
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("Duplicate JSON field: " + str(key))
            out[key] = value
        return out

    def constant(_):
        raise ValueError("Non-finite JSON number")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    canonical(value)
    return value


def seed(*parts) -> int:
    """Deterministic 60-bit seed from the study namespace and named parts."""
    value = STUDY + ":" + ":".join(map(str, parts))
    return int(hashlib.sha256(value.encode()).hexdigest()[:15], 16)


def write_new(path, value) -> None:
    """Write canonical JSON to a path that must not already exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(canonical(value) + "\n")
