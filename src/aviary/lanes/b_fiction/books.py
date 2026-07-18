"""Book acquisition + chunking. Pure functions, no LLM, no network.

Sources: local .txt (incl. Project Gutenberg dumps — boilerplate stripped),
.epub / .kepub (stdlib zipfile + html.parser; both are zipped XHTML — .kepub is
Kobo's epub), and .html (otwarchive-downloader output). Amazon .azw/.azw3/.kfx/
.mobi are a separate DRM-bearing family and are NOT read here. Book files live
outside the repo, listed in datagen/configs/lane_b.yaml.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

GUTENBERG_START = re.compile(r"\*{3} ?START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*{3}", re.I)
GUTENBERG_END = re.compile(r"\*{3} ?END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*{3}", re.I)

CHAPTER_HEADING = re.compile(
    r"^\s*(?:chapter|part|book)\s+(?:[0-9]+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|"
    r"nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty)\b.*$|^\s*[IVXLCDM]+\.?\s*$",
    re.I | re.MULTILINE,
)

_BLOCK_TAGS = {"p", "div", "br", "h1", "h2", "h3", "h4", "li", "blockquote", "tr"}
_SKIP_TAGS = {"script", "style", "head", "title"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_gutenberg(text: str) -> str:
    start = GUTENBERG_START.search(text)
    end = GUTENBERG_END.search(text)
    if start:
        text = text[start.end() :]
    if end:
        cut = GUTENBERG_END.search(text)
        if cut:
            text = text[: cut.start()]
    return text.strip()


def load_book_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return strip_gutenberg(path.read_text(encoding="utf-8", errors="replace"))
    if suffix in (".html", ".htm", ".xhtml"):
        return html_to_text(path.read_text(encoding="utf-8", errors="replace"))
    # .kepub is a Kobo EPUB — structurally a zipped-XHTML epub, same reader. (Amazon
    # .azw/.azw3/.kfx/.mobi are a different, DRM-bearing family and NOT handled here.)
    if suffix in (".epub", ".kepub"):
        return _load_epub(path)
    raise ValueError(f"unsupported book format: {path.name}")


def _load_epub(path: Path) -> str:
    chunks: list[str] = []
    with zipfile.ZipFile(path) as zf:
        names = [
            n
            for n in zf.namelist()
            if n.lower().endswith((".xhtml", ".html", ".htm"))
            and not any(skip in n.lower() for skip in ("toc", "nav", "cover", "copyright"))
        ]
        for name in sorted(names):
            text = html_to_text(zf.read(name).decode("utf-8", errors="replace"))
            if len(text) > 200:  # skip front-matter stubs
                chunks.append(text)
    return "\n\n".join(chunks)


@dataclass
class Chunk:
    idx: int
    text: str
    char_span: tuple[int, int] = (0, 0)


@dataclass
class BookConfig:
    work_id: str
    path: Path
    author: str = ""
    title: str = ""
    holdout: bool = False
    max_chunks: int | None = None
    extra: dict = field(default_factory=dict)


def chunk_text(text: str, target_chars: int = 16_000, overlap_paragraphs: int = 1) -> list[Chunk]:
    """Paragraph-aligned windows of ~target_chars (~4k tokens) with a small overlap so
    scenes cut at a boundary survive in the next chunk."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    current: list[str] = []
    size = 0
    for para in paragraphs:
        current.append(para)
        size += len(para)
        if size >= target_chars:
            chunks.append(Chunk(idx=len(chunks), text="\n\n".join(current)))
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            size = sum(len(p) for p in current)
    if current and (not chunks or size > target_chars // 4):
        chunks.append(Chunk(idx=len(chunks), text="\n\n".join(current)))
    return chunks
