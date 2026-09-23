"""Extract (someone speaks to Draco -> Draco replies) pairs from your own Harry Potter books.

The books are copyrighted: run this on copies you own, keep the output local
(``data/`` is git-ignored), and do not redistribute it.

Usage::

    python examples/sft/draco/extract_dialogue.py /path/to/harry.txt \
        --output examples/sft/draco/data/draco.jsonl

Accepts English ``.txt`` (one paragraph per line, UTF-8 or GB18030) and
``.epub``. Each paragraph is split into speech units (an adjacent ``"..." "..."``
pair means two speakers glued into one paragraph), and each unit is attributed
from its narration (``said Malfoy``, ``Harry snapped``). An unattributed unit
inherits the speaker of the unit two back when the unit in between is someone
else, which recovers the usual back-and-forth. A row is emitted when a unit by
someone other than Draco is directly followed by a Draco unit.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

_VERBS = (
    r"said|says|drawled|sneered|snarled|spat|jeered|called|shouted|yelled|asked|muttered|whispered|hissed|"
    r"snapped|replied|added|laughed|sniggered|scoffed|demanded|cried|growled|mocked|taunted|smirked|roared|"
    r"continued|went on|began|breathed|murmured|retorted|told|answered|repeated|insisted|exclaimed|bellowed|"
    r"panted|gasped|groaned|moaned|sighed|screamed|squeaked|barked|interrupted|urged|pleaded|agreed|protested"
)
_NAME = r"(?:(?:Professor|Mr\.|Mrs\.|Madam|Uncle|Aunt|Lord|Mad-Eye|Sir)\s+)?[A-Z][a-zA-Z']+(?:[- ][A-Z][a-zA-Z']+)?"
_SPEAKER_AFTER = re.compile(rf"\b(?:{_VERBS})\s+({_NAME})")
_SPEAKER_BEFORE = re.compile(rf"({_NAME})(?:,[^,\n]{{1,80}},)?\s+(?:{_VERBS})\b")  # also `Malfoy, who ..., said`
# Action beat with no speech verb: `Malfoy smirked. "..."` -- a name opening a narration sentence.
_BEAT_SUBJECT = re.compile(rf"(?:^|[.!?]\s+)({_NAME})\s+[a-z]", re.M)
_NOT_NAMES = {
    "He", "She", "They", "It", "I", "We", "You", "Then", "And", "But", "So", "Now", "When", "As", "The",
    "Someone", "Everyone", "Nobody", "Somebody",
}
_QUOTE_CHARS = str.maketrans({"“": '"', "”": '"'})
# A closing quote followed only by whitespace and a new opening quote starts another speaker.
_GLUED_SPEAKERS = re.compile(r'(?<=[.!?,—–-]")\s+(?=")')

DRACO = "Draco"


@dataclass
class Unit:
    words: str
    speaker: str | None
    index: int  # paragraph index, used to require adjacency


def canonical_speaker(name: str) -> str | None:
    words = name.split()
    while words and words[0] in _NOT_NAMES:  # "Then Harry said" -> "Harry"
        words.pop(0)
    name = " ".join(words)
    if not name or name.endswith("'s"):
        return None
    return DRACO if name in ("Draco", "Malfoy", "Draco Malfoy") else name


def attribute(narration: str) -> str | None:
    """Return the single speaker named in a unit's narration, else None."""

    names = {canonical_speaker(match.group(1)) for match in _SPEAKER_AFTER.finditer(narration)}
    names |= {canonical_speaker(match.group(1)) for match in _SPEAKER_BEFORE.finditer(narration)}
    names.discard(None)
    if not names:
        names = {canonical_speaker(match.group(1)) for match in _BEAT_SUBJECT.finditer(narration)}
        names.discard(None)
    return names.pop() if len(names) == 1 else None


def split_units(paragraph: str) -> list[tuple[str, str]]:
    """Split a paragraph into (spoken words, narration) speech units."""

    units = []
    for chunk in _GLUED_SPEAKERS.split(paragraph.translate(_QUOTE_CHARS)):
        if chunk.count('"') < 2 or chunk.count('"') % 2:
            continue  # no speech, or a speech that runs on into the next paragraph
        parts = chunk.split('"')
        words = " ".join(part.strip() for part in parts[1::2] if part.strip())
        words = words[:-1] + "." if words.endswith(",") else words  # `"Go away," said X` -> "Go away."
        narration = "\n".join(part.strip() for part in parts[0::2])
        if words:
            units.append((words, narration))
    return units


def speech_units(paragraphs: list[str]) -> list[Unit]:
    units = [
        Unit(words, attribute(narration), index)
        for index, paragraph in enumerate(paragraphs)
        for words, narration in split_units(paragraph)
    ]
    for k in range(2, len(units)):
        prev, before = units[k - 1], units[k - 2]
        alternating = units[k].index - before.index <= 2
        if units[k].speaker is None and alternating and prev.speaker and before.speaker not in (None, prev.speaker):
            units[k].speaker = before.speaker
    return units


def extract_rows(paragraphs: list[str], *, book: str, max_gap: int = 2) -> list[dict]:
    """Pair each Draco unit with the non-Draco unit right before it.

    ``max_gap`` is the largest paragraph distance between the two; 2 allows one
    pure-narration paragraph in between.
    """

    units = speech_units(paragraphs)
    rows = []
    for prev, unit in zip(units, units[1:]):
        adjacent = unit.index - prev.index <= max_gap
        if adjacent and unit.speaker == DRACO and prev.speaker not in (None, DRACO):
            rows.append({"book": book, "speaker": prev.speaker, "prompt_line": prev.words, "line": unit.words})
    return rows


class _ParagraphParser(HTMLParser):
    _BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "blockquote"}

    def __init__(self) -> None:
        super().__init__()
        self.paragraphs: list[str] = []
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in self._BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        self._buffer.append(data)

    def _flush(self) -> None:
        text = " ".join("".join(self._buffer).split())
        if text:
            self.paragraphs.append(text)
        self._buffer = []

    def close(self) -> None:
        super().close()
        self._flush()


def read_paragraphs(path: Path) -> list[str]:
    if path.suffix.lower() == ".epub":
        return _read_epub(path)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("gb18030")
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def _read_epub(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as book:
        container = ElementTree.fromstring(book.read("META-INF/container.xml"))
        opf_path = next(el.attrib["full-path"] for el in container.iter() if el.tag.endswith("rootfile"))
        opf = ElementTree.fromstring(book.read(opf_path))
        manifest = {el.attrib["id"]: el.attrib["href"] for el in opf.iter() if el.tag.endswith("}item")}
        spine = [el.attrib["idref"] for el in opf.iter() if el.tag.endswith("itemref")]
        base = posixpath.dirname(opf_path)
        paragraphs: list[str] = []
        for idref in spine:
            parser = _ParagraphParser()
            parser.feed(book.read(posixpath.join(base, manifest[idref])).decode("utf-8", errors="replace"))
            parser.close()
            paragraphs.extend(parser.paragraphs)
        return paragraphs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("books", nargs="+", type=Path, help="Your own .txt or .epub files.")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data" / "draco.jsonl")
    args = parser.parse_args()

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for path in args.books:
        book_rows = []
        for row in extract_rows(read_paragraphs(path), book=path.stem):
            key = (row["prompt_line"], row["line"])
            if key not in seen:
                seen.add(key)
                book_rows.append(row)
        print(f"{path.name}: {len(book_rows)} exchanges")
        rows.extend(book_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
