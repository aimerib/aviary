"""Exact + near-duplicate detection. Pure, dependency-free.

Exact: sha256 over normalized conversational text.
Near: MinHash (k=5 word shingles, 128 permutations) + LSH banding, Jaccard ~0.8.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict

from aviary.schema.records import ConversationRecord

_PERMS = 128
_SHINGLE_K = 5
_BANDS = 16  # 16 bands x 8 rows -> threshold ~ (1/16)^(1/8) ~= 0.71
_ROWS = _PERMS // _BANDS
_PRIME = (1 << 61) - 1

_rng = random.Random(0xA71A2)
_A = [_rng.randrange(1, _PRIME) for _ in range(_PERMS)]
_B = [_rng.randrange(0, _PRIME) for _ in range(_PERMS)]


def conversational_text(rec: ConversationRecord) -> str:
    parts = [
        f"{m.thought or ''} {m.content}" for m in rec.messages if m.role in ("user", "assistant")
    ]
    return normalize(" ".join(parts))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def exact_key(rec: ConversationRecord) -> str:
    return hashlib.sha256(conversational_text(rec).encode()).hexdigest()


def _shingles(text: str) -> set[int]:
    words = text.split()
    if len(words) < _SHINGLE_K:
        return {hash(" ".join(words)) & 0xFFFFFFFFFFFFFFF}
    out = set()
    for i in range(len(words) - _SHINGLE_K + 1):
        shingle = " ".join(words[i : i + _SHINGLE_K])
        out.add(int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest()))
    return out


def minhash(text: str) -> tuple[int, ...]:
    shingles = _shingles(text)
    return tuple(min((a * s + b) % _PRIME for s in shingles) for a, b in zip(_A, _B, strict=True))


def jaccard_estimate(sig1: tuple[int, ...], sig2: tuple[int, ...]) -> float:
    return sum(a == b for a, b in zip(sig1, sig2, strict=True)) / _PERMS


def find_duplicates(records: list[ConversationRecord], threshold: float = 0.8) -> dict[str, str]:
    """record_id -> id of the earlier record it duplicates (exact or near)."""
    dupes: dict[str, str] = {}
    seen_exact: dict[str, str] = {}
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    sigs: list[tuple[int, ...] | None] = []
    ids: list[str] = []

    for rec in records:
        rid = rec.provenance.record_id
        key = exact_key(rec)
        if key in seen_exact:
            dupes[rid] = seen_exact[key]
            sigs.append(None)
            ids.append(rid)
            continue
        seen_exact[key] = rid

        sig = minhash(conversational_text(rec))
        idx = len(sigs)
        candidates: set[int] = set()
        for band in range(_BANDS):
            bucket_key = (band, hash(sig[band * _ROWS : (band + 1) * _ROWS]))
            candidates.update(buckets[bucket_key])
            buckets[bucket_key].append(idx)
        for c in candidates:
            other = sigs[c]
            if other is not None and jaccard_estimate(sig, other) >= threshold:
                dupes[rid] = ids[c]
                break
        sigs.append(sig)
        ids.append(rid)
    return dupes
