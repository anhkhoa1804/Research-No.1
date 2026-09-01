# `datasets_vg150_clean` is not standard VG150 in its OBJECT vocabulary

Found while diagnosing an OOM in `runs/p37`, which tried to build a one-hot
matrix over the object labels and discovered there were 14,503 of them rather
than ~300.

## The measurement

```
VG150 object vocabulary (vocabulary/objects.json)   150 categories
distinct object labels in the p24 cache          16,929
object mentions in the cache                    271,756
  IN  the 150-category vocabulary               120,083  (44.2%)
  NOT in it                                     151,673  (55.8%)
```

Most frequent out-of-vocabulary objects: `wall` (3,250), `sky` (3,079),
`ground` (2,713), `grass` (2,479), `water` (1,893), `trees`, `clouds`, `leaves`,
`floor`, `pants`, `shadow`, `road`.

The **predicate** vocabulary is exactly the standard 50. Only the objects differ.

Restricting to relationships whose **both** endpoints are in the 150:

| | this dataset | VG150-restricted |
|---|---|---|
| train relationship rows | 1,046,427 | **293,376 (28.0%)** |
| distinct (s,o) groups, train | 212,981 | **8,392** |
| mean group size | 4.91 | **34.96** |

Standard VG150 preprocessing keeps the 150 most frequent object categories and
discards the rest. **This variant retains raw Visual Genome names**, so it holds
~3.6× more relationships than standard VG150 over a 113× larger object
vocabulary.

## What this does and does not affect

**Does NOT affect internal validity.** Every comparison in this programme —
PURE vs geometry vs probes vs nulls — is computed on identical rows of the same
cache. WPRD's prior-free property is *empirically verified* on exactly these
groups (`max |prior − group mean| = 9.4e-05`), and the frequency prior is keyed
on the same raw names, so conditioning on raw-name pairs cancels the prior that
is actually used. Conditioning on a **finer** partition also cancels any coarser
prior, so the construction is if anything stricter than category-level
conditioning.

**DOES affect external comparability.** `R@50 = 66.59` for a prior-only baseline
on this split is **not** directly comparable to published VG150 PredCls numbers.
Different object vocabulary, different pair population, different GT set. No
number in this project should be placed in a table beside published VG150
results without first restricting to the 150-category subset.

**DOES affect the cross-model study design.** A published checkpoint must be
compared on the VG150-restricted subset, not on this variant.

**DOES require correcting `runs/p50`** — see `docs/SUPERVISION_STRUCTURE_RESULT.md`
and `runs/p52`.

## Terminology correction

Throughout this project I wrote "(subject, object) **category** pair". The groups
are **raw object-name pairs**, which are finer. The WPRD construction and every
verification are unaffected; the wording was imprecise and is corrected here.
