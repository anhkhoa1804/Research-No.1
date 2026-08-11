# Evaluation pipeline

Traced from `openvocab_rel/evals.py`. All evaluation logic — PredCls,
SGCls, SGDet, calibration, pair-proposal diagnostics, role-swap
diagnostics, retrieval-adjacent grounding evals, the Grounding-DINO wrapper
— lives in this one file (~3861 lines, no internal module boundaries; kept
that way deliberately in the infrastructure cleanup pass — see the
cleanup's design notes on why splitting it was deferred rather than done).

## Protocol summary

| Protocol | Boxes | Labels | Notes |
|---|---|---|---|
| PredCls | GT, untouched | GT (only used to build triplet strings, never fed to the model) | `eval_sgg_use_gt_pairs=False` default means all N·(N−1) GT-box pairs are candidates, not just annotated ones — the flag name is about the *pair candidate set*, not the label source |
| PredCls-nogc | same | same | every predicate class scored per pair, no per-pair argmax constraint |
| SGCls | GT | CLIP zero-shot top-K classification of GT-box crops (default) | `eval_sgg_sgcls_oracle_labels` (default `False`) can silently substitute GT labels — always confirm this before citing SGCls numbers |
| SGDet | Grounding-DINO zero-shot detections, fully replace GT | detector's own predicted labels | matched to GT via IoU>=0.5 **and** exact label-triple match; `eval_sgg_grounding_dino_enabled` defaults `True` in config but shipped scripts default the env var to `false` (opt-in in practice) |
| Pair proposal / relationness | GT or detected | n/a | `gt_pair_recall@{32,64,96,128,256}` ("prop@K") kept as a diagnostic block architecturally separate from predicate-rank diagnostics |
| Retrieval | n/a | n/a | `retrieval.py`'s `TripletRetrievalIndex`/`build_triplet_records` are never called from `evals.py` — no retrieval eval is currently wired up despite the code existing |

## Scoring / calibration path

`_relation_predicate_logits` is the central function: computes raw
CLIP-text-cosine logits (always available) and classifier logits
(`predicate_logits`, or `calibrated_predicate_logits` if
`adaptive_calibration_enabled`), then combines per
`eval_sgg_predicate_score_mode` (`classifier` / `text` / `ensemble`
[default] / `auto`). Optional additions: eval-time frequency-prior/logit
adjustment (`freq_bias_*`, `logit_adj_tau` — both off by default),
relationness-score-based pair pruning (`eval_sgg_relationness_prune_k`).

**Raw vs. calibrated is a real distinction here, not cosmetic**: the raw
classifier path and any of the calibration/prior/ensemble paths produce
materially different numbers (see `README.md`'s own reporting policy: "Do
not mix raw and calibrated numbers in the same claim"). Always check which
path a reported number came from — `docs/known_issues.md` notes one place
(the README's LaTeX table snippet) where that policy isn't fully followed
in the docs themselves.

## Metrics

`_recall_from_matches` computes per-image recall (top-K truncation applied
per image, correctly). The **headline** `R@K`/`mR@K` fields aggregate hit
counts globally across the whole dataset before dividing (`sum(hits) /
sum(GT)`), which is a different statistic from the per-image-then-averaged
recall most VG150 literature reports under the same name — that
literature-standard variant is also computed, under the field name
`image_mean_R@K`. `mR@K` correctly excludes zero-occurrence predicate
classes from the averaging denominator (no NaN/skew risk). Background/
"relation" class logits are masked to `-10000` before ranking, so the
background class can never win the argmax.

## Pair-proposal vs. predicate-rank (Problem A vs. Problem B)

`pair_proposal_diag` (`gt_pair_recall@K` — does the pair survive pruning,
regardless of predicate correctness) and `pair_rank_diag` (rank of the
correct predicate *given* the pair survived, regardless of pruning) are two
separate diagnostic blocks. The headline R@K/mR@K still unavoidably
conflates both (a hit requires both), but this decomposition is what lets
you tell whether a regression came from losing pairs or misclassifying
predicates on surviving pairs.

## Role-swap diagnostic

`eval_sgg_standard`'s built-in role-swap check compares the logit for a
predicate on a pair's forward orientation against its reversed orientation,
restricted to non-symmetric predicates. A direction-blind model would score
0% (not 100%) on this test, so it isn't trivially gameable by a symmetric
representation — but it only tests sign, not magnitude, of the margin.
There is a second, standalone function (`eval_swap_consistency`) with a
known bug in its symmetric/asymmetric split — see `docs/known_issues.md`,
not fixed in this pass.

## Where to read more

- `docs/architecture/overview.md` — model architecture this pipeline scores
- `docs/architecture/data_flow.md` — end-to-end tensor flow for evaluation
- `docs/known_issues.md` — every gap/bug found in this pipeline, not fixed here
