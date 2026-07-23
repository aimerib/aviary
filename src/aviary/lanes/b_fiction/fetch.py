"""Source public-domain fiction for lane B from open repositories.

Project Gutenberg (via the Gutendex JSON API) is fully supported and needs no key;
Standard Ebooks has gated scripted access, so this refuses to write non-epub bytes
rather than feed lane B error pages named `.epub`.

Downloads go to a caller-chosen directory OUTSIDE the repo (data-in-git rule) and
are ingested by the normal lane B path: `.txt` has its Gutenberg boilerplate
stripped by `strip_gutenberg`, so there is nothing to convert.

The scoring/naming/validation/config logic is pure and offline-testable; only the
thin request layer below it touches the network (so the test suite's socket ban
covers everything that matters).
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import httpx

from aviary.lanes.b_fiction.books import BookConfig, strip_gutenberg

GUTENDEX = "https://gutendex.com/books/"
SE_INDEX = "https://standardebooks.org/ebooks/"
UA = "aviary-lane-b-fetcher/1.0 (private research corpus)"

# Mental-state / cognition / emotion words. Their density per 1000 words is a cheap
# proxy for interiority — the thing lane B extracts (inner thoughts, ToM). A
# plot-driven adventure runs low; close-third literary fiction runs high. This is a
# FIT test, not a quality one: a book that never names a feeling gives the
# dialogue-with-thoughts extractor nothing to work with. Calibrated against live
# Gutendex pulls (2026-07-23): Austen/Forster/Eliot ~11-12, Moby Dick 4.4
# (cetology), Monte Cristo 6.2 (adventure); ~7 is a sensible keep threshold.
_MENTAL = re.compile(
    r"\b(thought|thinks?|thinking|felt|feels?|feeling|wondered|wonder|realiz|realis"
    r"|remember|knew|believ|imagin|hoped?|feared?|wished?|doubt|suspect|consider"
    r"|sensed?|ached?|longed|dreaded|resent|worry|worried|understood|recogniz"
    r"|decided|ponder|mused|regret|ashamed|guilt|yearn|grief|angry|anxious)\w*",
    re.I,
)
_WORD = re.compile(r"[A-Za-z']+")


def interiority_score(text: str, sample_chars: int = 60_000) -> float:
    """Mental-state words per 1000 words over an early sample. Higher = more inner
    life. Returns 0.0 for texts too short to judge."""
    sample = text[:sample_chars]
    words = _WORD.findall(sample)
    if len(words) < 500:
        return 0.0
    return 1000.0 * len(_MENTAL.findall(sample)) / len(words)


def slugify(s: str, n: int = 60) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip()
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:n].strip("-").lower() or "untitled"


def is_epub(data: bytes) -> bool:
    """EPUBs are zip archives (PK magic). A gated site serving an HTML interstitial
    is not one — writing it as `.epub` would feed lane B an error page."""
    return data[:2] == b"PK"


def _author_name(raw: str) -> str:
    """Gutenberg stores 'Last, First'; lane B just wants a display string."""
    if "," in raw:
        last, first = (p.strip() for p in raw.split(",", 1))
        return f"{first} {last}".strip()
    return raw.strip()


def book_config(*, source: str, ident: str, path: Path, author: str, title: str) -> BookConfig:
    """A downloaded file as a lane B `BookConfig`. work_id is `<source>-<ident>` so
    reruns are stable and dedupe-able; holdout is left False for human review."""
    return BookConfig(
        work_id=slugify(f"{source}-{ident}", 80),
        path=path,
        author=_author_name(author),
        title=title.strip(),
    )


def percentiles(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {}
    s = sorted(scores)
    return {f"p{p}": s[min(len(s) - 1, p * len(s) // 100)] for p in (10, 25, 50, 75, 90)}


def dump_book_yaml(entries: list[BookConfig]) -> str:
    """Render entries as a `books:`-appendable YAML block. Hand-formatted (not
    yaml.dump) to match lane_b.yaml's quoted-path house style and stay reviewable."""
    lines = ["# append under `books:` in datagen/configs/lane_b.yaml — set holdout by hand"]
    for b in entries:
        lines.append(f"  - work_id: {b.work_id}")
        lines.append(f'    path: "{b.path}"')
        if b.author:
            lines.append(f'    author: "{b.author}"')
        lines.append(f'    title: "{b.title}"')
    return "\n".join(lines)


def existing_keys(lane_b_yaml: Path) -> set[str]:
    """(source ids + normalized titles) already in the corpus, to skip re-adding."""
    if not lane_b_yaml.exists():
        return set()
    import yaml

    raw = yaml.safe_load(lane_b_yaml.read_text()) or {}
    keys: set[str] = set()
    for b in raw.get("books", []):
        keys.add(str(b.get("work_id", "")).lower())
        keys.add(str(b.get("title", "")).strip().lower())
    keys.discard("")
    return keys


# --- network I/O (not unit-tested: the suite bans sockets) -----------------------


def _get(client: httpx.Client, url: str, *, params: dict | None = None, attempts: int = 5):
    """GET with backoff. Gutendex's /books/ query is intermittently slow (observed:
    two 45s timeouts, then an instant 200), so a transient failure retries rather
    than aborting a long run or silently dropping a good book."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            r = client.get(url, params=params)
            r.raise_for_status()
            return r
        except httpx.HTTPError as e:
            last = e
            time.sleep(min(2**i, 20))
    raise RuntimeError(f"gave up on {url} after {attempts} attempts: {last}")


def fetch_gutenberg(
    out: Path,
    *,
    limit: int,
    min_interiority: float,
    existing: set[str] | None = None,
    dry_run: bool = False,
    log=print,
) -> list[BookConfig]:
    """Pull popular English public-domain fiction, keep the interiority-rich, write
    `.txt` to `out`. Returns the kept BookConfigs (also emitted even on dry_run so
    a caller can print the score distribution before committing to downloads)."""
    out.mkdir(parents=True, exist_ok=True)
    existing = existing or set()
    have = {p.stem.split("--", 1)[0] for p in out.glob("gutenberg--*.txt")}
    kept: list[BookConfig] = []
    scores: list[float] = []
    page = 1
    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=45) as c:
        while len(kept) < limit:
            data = _get(
                c,
                GUTENDEX,
                params={
                    "languages": "en",
                    "copyright": "false",
                    "mime_type": "text/plain",
                    "topic": "fiction",
                    "sort": "popular",
                    "page": page,
                },
            ).json()
            if not data.get("results"):
                break
            for b in data["results"]:
                if len(kept) >= limit:
                    break
                gid = str(b["id"])
                if gid in have or f"gutenberg-{gid}" in existing:
                    continue
                fmts = b.get("formats", {})
                url = next(
                    (u for k, u in fmts.items() if k.startswith("text/plain") and "utf-8" in k),
                    next((u for k, u in fmts.items() if k.startswith("text/plain")), None),
                )
                if not url or url.endswith(".zip"):
                    continue
                author = b["authors"][0]["name"] if b.get("authors") else "Unknown"
                title = b.get("title", "untitled")
                time.sleep(2.0)  # be kind to gutenberg.org
                try:
                    text = strip_gutenberg(_get(c, url).text)
                except RuntimeError as e:
                    log(f"  skip {gid}: {e}")
                    continue
                score = interiority_score(text)
                scores.append(score)
                if score < min_interiority:
                    continue
                path = out / f"gutenberg--{gid}--{slugify(author, 30)}--{slugify(title)}.txt"
                if not dry_run:
                    path.write_text(text, encoding="utf-8")
                kept.append(
                    book_config(
                        source="gutenberg", ident=gid, path=path, author=author, title=title
                    )
                )
                log(f"  [{len(kept)}/{limit}] interiority={score:4.1f}  {title} — {author}")
            page += 1
    if scores:
        log(
            f"\nscored {len(scores)} | kept {len(kept)} | interiority percentiles: "
            + "  ".join(f"{k}={v:.1f}" for k, v in percentiles(scores).items())
        )
    return kept


def fetch_standardebooks(out: Path, *, limit: int, log=print) -> list[BookConfig]:
    """Standard Ebooks gated scripted access (OPDS 401; epub URLs serve an XHTML
    interstitial). This does NOT scrape around that: it validates PK-zip bytes and,
    the moment it gets HTML, reports the gate and stops. The legit route is patron
    OPDS access — Gutenberg already covers the volume, so this is a bonus pass."""
    out.mkdir(parents=True, exist_ok=True)
    kept: list[BookConfig] = []
    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=45) as c:
        try:
            idx = _get(c, SE_INDEX).text
        except RuntimeError as e:
            log(f"Standard Ebooks index unreachable: {e}")
            return kept
        for slug in dict.fromkeys(re.findall(r'href="(/ebooks/[^"]+/[^"/]+)"', idx)):
            if len(kept) >= limit:
                break
            author_slug, title_slug = slug.split("/")[-2:]
            url = f"https://standardebooks.org{slug}/downloads/{author_slug}_{title_slug}.epub"
            time.sleep(1.5)
            try:
                blob = _get(c, url).content
            except RuntimeError:
                continue
            if not is_epub(blob):
                log(
                    "\nStandard Ebooks is serving HTML, not epubs — scripted download is "
                    "gated.\nBecome a patron for OPDS access and point this at the authed "
                    "feed, or\npull a handful by hand. Gutenberg already covers the volume."
                )
                return kept
            path = out / f"standardebooks--{slugify(slug.replace('/', '-'))}.epub"
            path.write_bytes(blob)
            kept.append(
                book_config(
                    source="standardebooks",
                    ident=title_slug,
                    path=path,
                    author=author_slug.replace("-", " "),
                    title=title_slug.replace("-", " "),
                )
            )
            log(f"  [{len(kept)}/{limit}] {slug}")
    return kept
