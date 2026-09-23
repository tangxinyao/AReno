from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "sft" / "draco"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"sft_draco_{name}_for_tests", EXAMPLE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_english_attribution_keeps_only_draco():
    extract = _load("extract_dialogue")

    assert extract.draco_line("“Nice robes,” drawled Malfoy. “Second-hand?”", "en") == "Nice robes, Second-hand?"
    assert extract.draco_line("Malfoy sneered. ‘Don’t be slow, Crabbe.’", "en") == "Don’t be slow, Crabbe."
    assert extract.draco_line("“Enough,” said Lucius Malfoy.", "en") is None
    assert extract.draco_line("“Go away,” said Harry, glaring at Malfoy.", "en") is None
    assert extract.draco_line("Malfoy walked past without a word.", "en") is None


def test_chinese_attribution_keeps_only_draco():
    extract = _load("extract_dialogue")

    assert extract.draco_line("马尔福拖着长腔说：“新长袍不错。”", "zh") == "新长袍不错。"
    assert extract.draco_line("马尔福先生冷冷地说：“够了。”", "zh") is None
    assert extract.draco_line("哈利瞪着马尔福说：“走开。”", "zh") is None


def test_extract_rows_from_epub_uses_preceding_context(tmp_path):
    extract = _load("extract_dialogue")
    book = tmp_path / "book.epub"
    with zipfile.ZipFile(book, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<container><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>',
        )
        archive.writestr(
            "OEBPS/content.opf",
            '<package xmlns="http://www.idpf.org/2007/opf"><manifest>'
            '<item id="c1" href="c1.xhtml"/></manifest><spine><itemref idref="c1"/></spine></package>',
        )
        archive.writestr(
            "OEBPS/c1.xhtml",
            "<html><body><p>The corridor was cold.</p><p>“Lost again?” sneered Malfoy.</p></body></html>",
        )

    paragraphs = extract.read_paragraphs(book)
    rows = extract.extract_rows(paragraphs, lang="en", book="book", context_paras=4, context_chars=1500)

    assert rows == [{"book": "book", "lang": "en", "context": "The corridor was cold.", "line": "Lost again?"}]


def test_draco_loader_returns_prompt_response_rows():
    loader = _load("dataset_loader")
    raw = [{"lang": "en", "context": "The corridor was cold.", "line": "Lost again?"}]

    records = loader.load_training_dataset("unused", default_loader=lambda _: raw)

    assert records[0]["response"] == "Lost again?"
    assert "You are Draco Malfoy" in records[0]["prompt"]
    assert "The corridor was cold." in records[0]["prompt"]
