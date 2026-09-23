"""Extract Draco Malfoy's lines from your own Harry Potter ebooks into SFT rows.

The books are copyrighted: run this on copies you own, keep the output local
(``data/`` is git-ignored), and do not redistribute it.

Usage::

    python examples/sft/draco/extract_dialogue.py books/*.epub \
        --output examples/sft/draco/data/draco.jsonl

Accepts ``.txt`` (one paragraph per line or blank-line separated) and ``.epub``.
English and Chinese (simplified or traditional) editions are both supported;
the language is detected per book. Attribution is heuristic and favors
precision: a paragraph is kept only when its narration names Malfoy/Draco next
to a speech verb and names no other speaker.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

# English -----------------------------------------------------------------
_EN_VERBS = (
    r"said|drawled|sneered|snarled|spat|jeered|called|shouted|yelled|asked|muttered|whispered|hissed|"
    r"snapped|replied|added|laughed|sniggered|scoffed|demanded|cried|growled|mocked|taunted|smirked|"
    r"continued|went on|began|breathed|murmured|retorted|jibed"
)
_EN_DRACO = r"(?:Draco(?:\s+Malfoy)?|Malfoy)"
_EN_DRACO_SPEAKS = re.compile(rf"\b(?:{_EN_VERBS})\s+{_EN_DRACO}\b|\b{_EN_DRACO}\s+(?:{_EN_VERBS})\b")
_EN_OTHER_SPEAKS = re.compile(rf"\b(?:{_EN_VERBS})\s+([A-Z][a-z]+)|\b([A-Z][a-z]+)\s+(?:{_EN_VERBS})\b")
# Other Malfoys are masked before attribution so they never count as Draco.
_EN_OTHER_MALFOYS = re.compile(r"\b(?:Lucius|Narcissa|Mr\.?|Mrs\.?)\s+Malfoy\b")
_EN_NOT_NAMES = {"He", "She", "They", "It", "I", "We", "You", "Draco", "Malfoy", "Then", "And", "But", "Someone"}
_EN_DOUBLE_QUOTE = re.compile(r"[“\"]([^”\"]+)[”\"]")
# Closing ’ followed by a letter is an apostrophe (don’t), not a closing quote.
_EN_SINGLE_QUOTE = re.compile(r"‘(.+?)’(?![A-Za-z])")

# Chinese: 人民文学版 马尔福/德拉科, 皇冠版 馬份/跩哥 ---------------------------
_ZH_VERBS = r"说|說|道|问|問|叫|喊|嚷|笑|吼|讥|譏"
_ZH_DRACO = r"(?:德拉科|跩哥|马尔福|馬份)"
_ZH_DRACO_SPEAKS = re.compile(rf"{_ZH_DRACO}[^“”「」。！？]{{0,10}}?(?:{_ZH_VERBS})")
_ZH_OTHER_MALFOYS = re.compile(r"卢修斯·?马尔福|盧修斯·?馬份|纳西莎·?马尔福|水仙·?馬份|马尔福(?:先生|夫人)|馬份(?:先生|夫人)")
_ZH_OTHERS = (
    r"哈利|罗恩|榮恩|赫敏|妙麗|海格|邓布利多|鄧不利多|斯内普|石內卜|麦格|麥教授|纳威|奈威|金妮|弗雷德|乔治|"
    r"克拉布|高尔|潘西|卢娜|小天狼星|卢平|路平|穆迪|伏地魔|贝拉特里克斯|韦斯莱|衛斯理"
)
_ZH_OTHER_SPEAKS = re.compile(rf"(?:{_ZH_OTHERS})[^“”「」。！？]{{0,10}}?(?:{_ZH_VERBS})")
_ZH_QUOTE = re.compile(r"[“「]([^”」]+)[”」]")

_BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "blockquote"}


class _ParagraphParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paragraphs: list[str] = []
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in _BLOCK_TAGS:
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
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return [line.strip() for line in text.splitlines() if line.strip()]


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


def detect_language(paragraphs: list[str]) -> str:
    sample = "".join(paragraphs[:500])
    cjk = sum(1 for char in sample if "一" <= char <= "鿿")
    return "zh" if cjk > len(sample) * 0.3 else "en"


def draco_line(paragraph: str, lang: str) -> str | None:
    """Return Draco's spoken words in a paragraph, or None if he is not the sole speaker."""

    if lang == "zh":
        quotes = _ZH_QUOTE.findall(paragraph)
        narration = _ZH_OTHER_MALFOYS.sub("其他人", _ZH_QUOTE.sub("", paragraph))
        if not quotes or not _ZH_DRACO_SPEAKS.search(narration) or _ZH_OTHER_SPEAKS.search(narration):
            return None
        return "".join(quote.strip() for quote in quotes)

    quote_re = _EN_SINGLE_QUOTE if paragraph.count("‘") > paragraph.count("“") else _EN_DOUBLE_QUOTE
    quotes = quote_re.findall(paragraph)
    narration = _EN_OTHER_MALFOYS.sub("Other", quote_re.sub(" ", paragraph))
    if not quotes or not _EN_DRACO_SPEAKS.search(narration):
        return None
    for match in _EN_OTHER_SPEAKS.finditer(narration):
        name = match.group(1) or match.group(2)
        if name not in _EN_NOT_NAMES:
            return None
    return " ".join(quote.strip() for quote in quotes)


def extract_rows(paragraphs: list[str], *, lang: str, book: str, context_paras: int, context_chars: int) -> list[dict]:
    rows = []
    for index, paragraph in enumerate(paragraphs):
        line = draco_line(paragraph, lang)
        if not line or len(line) < 2:
            continue
        context = "\n".join(paragraphs[max(0, index - context_paras) : index])[-context_chars:]
        if context:
            rows.append({"book": book, "lang": lang, "context": context, "line": line})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("books", nargs="+", type=Path, help="Your own .txt or .epub files.")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data" / "draco.jsonl")
    parser.add_argument("--context-paras", type=int, default=4, help="Preceding paragraphs used as the scene.")
    parser.add_argument("--context-chars", type=int, default=1500, help="Keep at most this many trailing chars.")
    args = parser.parse_args()

    rows: list[dict] = []
    seen: set[str] = set()
    for path in args.books:
        paragraphs = read_paragraphs(path)
        lang = detect_language(paragraphs)
        book_rows = extract_rows(
            paragraphs, lang=lang, book=path.stem, context_paras=args.context_paras, context_chars=args.context_chars
        )
        book_rows = [row for row in book_rows if not (row["line"] in seen or seen.add(row["line"]))]
        print(f"{path.name}: {lang}, {len(paragraphs)} paragraphs, {len(book_rows)} Draco lines")
        rows.extend(book_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
