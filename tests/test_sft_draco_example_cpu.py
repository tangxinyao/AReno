from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "sft" / "draco"


def _load(name: str):
    module_name = f"sft_draco_{name}_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses need the module registered
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_attribution_by_speech_verb_beat_and_apposition():
    extract = _load("extract_dialogue")

    assert extract.attribute("\n drawled Malfoy.") == "Draco"
    assert extract.attribute("Then Harry said,\n") == "Harry"
    assert extract.attribute("Malfoy smirked.\n") == "Draco"
    assert extract.attribute("Malfoy, who had turned around, said,\n") == "Draco"
    assert extract.attribute("\n said Lucius Malfoy.") == "Lucius Malfoy"
    assert extract.attribute("\n he said.") is None


def test_split_units_separates_glued_speakers():
    extract = _load("extract_dialogue")

    units = extract.split_units('"Give it back," Ron called. "Now!" "Make me," said Malfoy.')

    assert [words for words, _ in units] == ["Give it back, Now!", "Make me."]


def test_extract_rows_pairs_other_speaker_with_draco_reply():
    extract = _load("extract_dialogue")
    paragraphs = [
        '"Lost again?" said Ron.',
        "The corridor was cold.",
        '"Not as lost as you," Malfoy sneered.',
        '"Leave it," said Harry.',
        '"Why should I?"',  # unattributed: alternation makes it Draco's
    ]

    rows = extract.extract_rows(paragraphs, book="book")

    assert [(row["speaker"], row["prompt_line"], row["line"]) for row in rows] == [
        ("Ron", "Lost again?", "Not as lost as you."),
        ("Harry", "Leave it.", "Why should I?"),
    ]


def test_read_paragraphs_handles_gb18030_txt_and_epub(tmp_path):
    extract = _load("extract_dialogue")
    txt = tmp_path / "book.txt"
    txt.write_bytes('　　"Hello," said Harry.\n'.encode("gb18030"))
    assert extract.read_paragraphs(txt) == ['"Hello," said Harry.']

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
        archive.writestr("OEBPS/c1.xhtml", "<html><body><p>One.</p><p>Two.</p></body></html>")
    assert extract.read_paragraphs(book) == ["One.", "Two."]


def test_draco_loader_returns_prompt_response_rows():
    loader = _load("dataset_loader")
    raw = [{"speaker": "Ron", "prompt_line": "Lost again?", "line": "Not as lost as you."}]

    records = loader.load_training_dataset("unused", default_loader=lambda _: raw)

    assert records == [{"prompt": "Ron: Lost again?", "response": "Not as lost as you."}]
