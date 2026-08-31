# `pair_logit_dump_v2` — cache schema for experiment C′

Written by `openvocab_rel/evals.py::_write_pair_dump`, enabled only by
`--eval_sgg_dump_pair_logits_path`. Default off; when off, no call site runs.

**Purpose.** C′ must cost exactly **one** GPU pass. Everything downstream — the
τ sweep, the α sweep, the Pareto placement, the top-5 rescue accounting, the
uncertainty strata, the null — is then CPU work over this file. Nothing in it
is a metric; it is the raw material metrics are derived from.

Validate with `tools/validate_pair_dump.py` **before** any analysis reads it.

---

## 1. Top-level scalars and metadata

| Key | Type | Meaning |
|---|---|---|
| `schema` | str | `"pair_logit_dump_v2"` |
| `schema_doc` | str | path to this file |
| `composition` | str | the exact formula the evaluator applied, in words |
| `pred_vocab` | list[str] | **runtime** predicate vocabulary, in index order. 51 entries: VG150's 50 + one background class |
| `background_predicate_indices` | list[int] | columns masked to −1e4 *after* composition |
| `predicate_alias_map` | dict[str,str] | the evaluator's own alias map, entries that change a label. Only `near→next to` and `wears→wearing` fire on VG150's 50 |
| `n_images`, `n_pairs` | int | totals, cross-checked by the validator |
| `freq_bias_alpha` | float | α **recorded but NOT applied** to `prior_rows` |
| `bayes_calibration_weight` | float | overrides α when non-zero (see `_apply_frequency_bias`) |
| `eval_freq_bias_tau` | float | τ **recorded; must be 0.0** — rows are stored raw |
| `ensemble_alpha` | float | configured ensemble weight |
| `ensemble_alpha_used` | float | the weight actually applied (differs under `mode="auto"`) |
| `branch` | str | `ensemble` / `text_only` / `classifier_only` |
| `classifier_temperature`, `text_temperature` | float | ensemble temperatures |
| `adaptive_calibration_enabled`, `eval_logit_adj_tau`, `score_mode` | | protocol flags |
| `use_gt_pairs`, `iou_thresh`, `use_vg_aliases` | | protocol flags |
| `freq_bias_path`, `resume_from`, `vg150_root` | str | artifact identity |
| `missing_prior_images`, `missing_text_logits`, `missing_cls_logits` | int | must be 0 |

## 2. Per-image lists

Every key below is a list of length `n_images`, index-aligned. For image `i`
with `n = pairs[i].shape[0]` pairs and `P = len(pred_vocab)`:

| Key | Shape / type | Meaning |
|---|---|---|
| `image_id` | str | dataset image id |
| `pairs` | `LongTensor[n, 2]` | `(subject_object_index, object_object_index)` into `obj_labels[i]` |
| `pair_index` | `LongTensor[n]` | `0..n-1`; makes row identity explicit |
| `obj_labels` | list[str] | GT object class per object slot, **raw** |
| `subj_label` | list[str], len n | `obj_labels[i][pairs[i][:,0]]`. Redundant **by construction**; the validator re-derives and asserts equality, so it is a checked invariant |
| `obj_label` | list[str], len n | `obj_labels[i][pairs[i][:,1]]`, same |
| `obj_boxes` | `FloatTensor[n_obj, 4]` | GT boxes |
| `model_logits` | `FloatTensor[n, P]` | the composed model term, **before** prior composition |
| `text_logits` | `FloatTensor[n, P]` | the text/cosine branch, pre-normalisation |
| `cls_logits` | `FloatTensor[n, P]` | the predicate-classifier branch, post `_apply_eval_logit_adjustment`, pre-normalisation |
| `prior_rows` | `FloatTensor[n, P]` | `log P(p \| s, o)` — **raw**: no τ, no α |
| `gt_subj_idx`, `gt_obj_idx` | list[int] | GT triplet endpoints, into `obj_labels[i]` |
| `gt_pred` | list[str] | GT predicate labels, **RAW — never alias-normalised** |
| `gt_subj_label`, `gt_obj_label` | list[str] | GT endpoint classes, raw |

### Why `gt_pred` must be raw

The evaluator normalises GT through `_normalize_triplet_labels` before scoring,
which collapses `near→next to` and `wears→wearing` and turns the 50-class VG150
vocabulary into 48. That transformation is **lossy and irreversible**. If the
cache stored normalised GT, the 50-class arm would be unrecoverable from a
60-GPU-minute artifact.

The cache therefore stores raw GT and exports the alias map, so the CPU side can
produce **either** scheme from **one** cache. Pinned by
`tests/test_pair_logit_dump.py::test_dump_stores_raw_gt_predicates_not_aliased_ones`.

### Why both branches are stored

Under the historical protocol `ensemble_alpha = 0.0`, so

    model_logits  ==  normalise(text_logits) / text_temperature

exactly, and `cls_logits` — the trained predicate classifier, including adaptive
calibration — is multiplied by zero and discarded. That is the single point in
the pipeline where model signal is most obviously suppressed. Storing the
discarded branch is the only way C′ can ask whether it carries information the
surviving branch does not, without a second GPU pass. Pinned by
`test_historical_protocol_discards_the_classifier_branch_entirely`.

## 3. Derived quantities (NOT stored — computed on CPU, deterministically)

Storing these would duplicate state that can drift. Each is a pure function of
the fields above; `tools/validate_pair_dump.py` recomputes the composition
(check S10) to prove the stored tensors are self-consistent.

```
prior_tau[i]   = prior_rows[i] - tau * global_log_prob            # global_log_prob = log P(p)
combined[i]    = model_logits[i] + alpha * prior_tau[i]
masked         = combined with background columns set to -1e4
probs[i]       = softmax(masked, dim=-1)
rank[i][:, p]  = position of p when masked[i] is sorted descending
topK_ids[i]    = argsort(masked[i], descending=True)[:, :K]
prediction[i]  = topK_ids[i][:, 0]
```

`global_log_prob` is the prior file's own class marginal, remapped onto the
runtime vocabulary by `_load_frequency_bias` — the same tensor
`_apply_freq_bias_tau` uses, so the CPU τ and the in-evaluator τ are the same
transformation.

## 4. Validation gate

`tools/validate_pair_dump.py` must print **CACHE VALID** (exit 0) before any
analysis runs. It re-implements every check from this document independently of
the analysis tool, so a bug in one cannot mask a bug in the other. Checks
S1–S12 are listed in that tool's docstring; S7/S8/S9 are the ones that protect
the denominator, S10 the one that protects the composition.
