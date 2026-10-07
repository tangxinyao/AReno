# NOTICE

This directory contains an independent Python reimplementation of Slay the
Spire 2 run and combat mechanics for the Jev decision-scoring policy. It is a
clean-room implementation inside the AReno repository; no decompiled game
source or binary assets are vendored.

## Game content

*Slay the Spire 2* is a trademark of MegaCrit, LLC. The simulator here aims to
reproduce only the mechanical rules (cost, damage, draw/discard flow, enemy
intents, map/shop/event structure) at the level needed to train an RL policy.
No art, audio, flavor text, decompiled source, or save-file schemas are
redistributed with this code.

## Reference implementations

The code structure, test strategy, and some numeric constants are informed by
reading the following public reimplementations:

- [r33hab/sts2](https://github.com/r33hab/sts2) — C# NativeAOT emulator (MIT).
  Primary port oracle for monster movesets, card damage curves, and relic
  trigger timing.
- [yizhang-dream/sts2-simulator](https://github.com/yizhang-dream/sts2-simulator)
  — Pure Python simulator with a live-game TCP bridge (research use notice;
  structure referenced only, no content copied).
- [HIX4123/sts2-simulator](https://github.com/HIX4123/sts2-simulator) — Pure
  Python simulator (no explicit license; used for cross-check on specific
  mechanics only).

Where this code borrows a non-trivial algorithmic shape (e.g. hook dispatch
ordering, RNG stream layout) from one of the above, the file-level docstring
notes which project the pattern came from.
