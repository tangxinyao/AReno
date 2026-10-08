# NOTICE

This directory contains an independent Python reimplementation of Slay the
Spire 2 combat mechanics for the Jev decision-scoring policy. No decompiled
game source or binary assets are vendored.

## Game content

*Slay the Spire 2* is a trademark of MegaCrit, LLC. The data under `data/`
records game facts for STS2 public build v0.107.1 (Steam build 23811903):
card and monster names, costs, rarities, targets, keywords, numeric values,
monster move sets and encounter pools, plus the short rules text printed on
cards (`description`). The rules text is kept because the live game, read
through STS2MCP, shows the same text to the policy; matching it keeps sim
and live prompts aligned. No art, audio, lore/flavor text or save-file
schemas are redistributed.

## Sources

- [r33hab/sts2](https://github.com/r33hab/sts2) (MIT), commit b745217, built
  against the same game build. Its emulator (`Core/CombatEngine.cs`,
  `Core/EnemyAI.cs`, `Core/Effects/CardEffects.cs`, `Core/BuffSystem.cs`,
  `Core/CombatFactory.cs`, `Generated/*.g.cs`) is the behavior reference for
  this port: turn order, damage/block math, power timings, monster AI,
  encounter rosters and pools. Monster/card numbers were cross-checked
  against its generated tables.
- [Spire Codex](https://github.com/ptrlrd/spire-codex) data for v0.107.1
  (`data/eng/*.json`, extracted from the game's localization and models)
  was used as a cross-check for names, rules text, numbers and upgrade
  values while the JSON here was written. No Spire Codex files or code are
  included in this repository.
- [yizhang-dream/sts2-simulator](https://github.com/yizhang-dream/sts2-simulator)
  and [HIX4123/sts2-simulator](https://github.com/HIX4123/sts2-simulator) —
  structure referenced only, no content copied.

`data/sts2_card_ids.json` is a list of STS2 card identifiers, derived from
r33hab/sts2's `data/id_map.json` and `data/card_id_classes.json` (MIT) at
commit b745217.
