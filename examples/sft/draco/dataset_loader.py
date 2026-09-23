"""Dataset loader for Draco Malfoy role-play SFT rows from extract_dialogue.py."""

from __future__ import annotations

_PROMPTS = {
    "en": (
        "You are Draco Malfoy, a Slytherin student at Hogwarts. Stay in character.\n\n"
        "Scene:\n{context}\n\n"
        "Reply with only what Draco says next."
    ),
    "zh": ("你是德拉科·马尔福，霍格沃茨的斯莱特林学生。保持角色。\n\n场景：\n{context}\n\n只回复德拉科接下来说的话。"),
}


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Turn {lang, context, line} rows into prompt/response pairs."""

    records = []
    for row in default_loader(dataset_path):
        prompt = _PROMPTS[row.get("lang") or "en"].format(context=str(row["context"]).strip())
        records.append({"prompt": prompt, "response": str(row["line"]).strip()})
    return records
