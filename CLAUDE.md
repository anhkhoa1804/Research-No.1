# Research-No.1 — Claude Operating Rules

## Mission

This is a research codebase for Scene Graph Generation (SGG),
currently focused on VG150 / PredCls and open-vocabulary relational modeling.

The goal is scientifically defensible research, not merely a higher metric.

## Research discipline

Before changing research logic:

1. Establish current truth from source code, tests, configuration, and artifacts.
2. Distinguish:
   - VERIFIED FACT
   - MEASURED RESULT
   - INFERENCE
   - HYPOTHESIS
3. Do not present inference as evidence.
4. Prefer falsification over confirmation.
5. Define success/failure criteria before expensive experiments.
6. Prefer the cheapest decisive experiment before GPU-heavy experiments.

Previous agents may be wrong. Challenge previous conclusions when evidence supports doing so.

## Repository safety

- Never modify historical checkpoint weights.
- Never silently replace historical artifacts with newly generated ones.
- Never casually reorder the canonical predicate vocabulary.
- Never add datasets, checkpoints, caches, or large generated artifacts to git.
- Keep experiment outputs under runs/.
- Never delete historical experiment artifacts without explicit justification.
- Never rewrite git history unless explicitly instructed.
- Never push to main automatically.
- Always inspect git diff before committing.
- Always run relevant tests after code changes.

## Dataset safety

Treat the original dataset archive as immutable evidence.

Current raw archive:
~/VG150_dataset.zip

Current extracted dataset:
~/VG150_dataset_extract/

Do NOT:
- delete or rewrite the raw ZIP;
- repair images in place;
- silently alter JSONL;
- silently change train/validation/test membership;
- silently regenerate vocabulary.

Any repaired or transformed dataset must use a new versioned directory and be documented.

The historical frequency prior and the current train-derived frequency prior are different scientific artifacts.

Historical:
checkpoints/demo_best/frequency_prior.json

Current train-derived:
datasets_vg150_clean/frequency_prior_train.json

Never substitute one for the other without explicit justification.

## Historical checkpoint safety

Never modify:

checkpoints/demo_best/pure_best_adapt_light_mR50.pt
checkpoints/demo_best/frequency_prior.json
checkpoints/demo_best/demo_config.env

Verify SHA256 before any historical reproduction.

## Compute

This machine is a GCP NVIDIA L4 VM.

Expected environment:
- NVIDIA L4
- 24 GB VRAM
- CUDA available
- PyTorch CUDA build

Use GPU deliberately.

Before an expensive run, record:
- git commit
- git status
- environment
- dataset identity
- checkpoint identity
- configuration
- output directory

Do not launch multi-hour training without an explicit research plan and pilot.

## Autonomous work

Claude may:
- inspect the repository;
- inspect dataset metadata;
- inspect experiment artifacts;
- edit code;
- write tests;
- run tests;
- run bounded experiments;
- perform static analysis;
- create documentation;
- create new research branches;
- commit changes.

Claude must NOT:
- push to main;
- modify historical checkpoints;
- destroy raw dataset evidence;
- silently alter split membership;
- launch long training without approval;
- silently change scientific defaults.

## Autonomous Research Execution Policy

- Safe infrastructure, tests, diagnostics, dataset verification, documentation, and debugging may be executed autonomously.
- Scientific/model changes may be implemented autonomously only after the research hypothesis and success/failure criteria are explicitly stated in the current response.
- Never modify historical artifacts.
- Never modify the raw dataset.
- Never edit a reproducibility manifest merely to make a check pass.
- Never launch an expensive experiment if a cheaper decisive control has not been considered.
- Use the GPU efficiently: batch operations, cache reusable features, avoid redundant extraction, use mixed precision where scientifically safe, and avoid repeated model loading.
- For every GPU experiment record GPU model, VRAM, git commit, configuration, dataset identity, checkpoint identity, start/end time, and output path.
- For long-running jobs prefer resumable execution and persistent logs.
- Before a full run, perform the cheapest meaningful smoke/pilot check.
- You may challenge previous agent conclusions and may recommend substantial architecture changes.
- Do not optimize for a prettier result; optimize for scientifically defensible evidence.

## Current priority

The immediate priority is NOT training.

First:
1. audit repository;
2. audit dataset;
3. verify environment;
4. run tests;
5. run preflight;
6. identify real scientific bottlenecks;
7. evaluate architecture options;
8. design decisive experiments;
9. only then consider GPU training.

## Architectural freedom

Do not assume the current architecture must be preserved.

You may propose:
- simplification;
- refactoring;
- new prediction formulations;
- new heads;
- new objectives;
- candidate generation/ranking changes;
- alternative losses;
- architectural removal;
- architectural replacement;
- task/evaluation reformulation.

But every scientific change must be tied to measured evidence.

## Expensive experiment rule

For a new research idea:

1. static/unit validation;
2. synthetic or tiny smoke test;
3. bounded real-data pilot;
4. only then full experiment.

Record why each stage passed.

## Git workflow

Preferred:
feature/research branch
→ tests
→ review
→ commit

Do not modify main automatically.
