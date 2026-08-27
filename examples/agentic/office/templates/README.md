# Office template catalog

`catalog.json` is consumed by `../dataset_generator.py` through
`--template-dir`. It contains task structure and field constraints, not fixed
instance values.

- `filename` fields generate a fresh deterministic random stem and preserve the
  required extension.
- `text` fields generate deterministic random text at the configured length.
- Prompt bodies are rendered through a compositional grammar with prose,
  numbered, bulleted, goal/constraint, operation-chain, and bracketed layouts.
- The fixture, prompt, oracle, and grader all consume the same rendered spec.
- A record is emitted only when its generated oracle artifact receives score
  `1.0`.

The source metadata records all 25 retry-correct and 21 wrong deliverable case
IDs used to define coverage. `risk_tags` preserve the observed failure modes
without copying fixed source prompts or filenames.
