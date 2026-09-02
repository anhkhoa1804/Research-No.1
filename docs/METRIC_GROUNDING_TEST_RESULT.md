# `runs/p61` — the rank inversion REPLICATES on the held-out TEST split

`p49`/`p53` on the validation split; this is the same twelve arms, the same
composition, tau=0, computed on the `p54` TEST cache. Registered as items 4 and
5 of `docs/TEST_SPLIT_REPLICATION_PREREGISTRATION.md`.

| arm | R@50 | mR@50 | Pareto | **WPRD** |
|---|---|---|---|---|
| pair prior only | 66.636 | 22.027 | +0.000 | **0.5000** |
| random null | 64.975 | 21.166 | −5.533 | 0.5063 |
| PURE text (α=0, deployed) | 67.013 | **22.401** | +0.374 | 0.5446 |
| PURE ens α=0.25 | 67.204 | 22.306 | +0.278 | 0.5521 |
| PURE ens α=0.5 | 67.249 | 22.029 | +0.001 | 0.5614 |
| PURE ens α=0.75 | 67.249 | 21.660 | −0.367 | 0.5691 |
| PURE classifier (α=1) | 67.169 | 21.496 | −0.531 | 0.5712 |
| geometry linear | 67.383 | **18.059** | −3.968 | 0.5897 |
| geometry MLP | 67.433 | 19.870 | −2.158 | 0.6060 |
| text + geometry | 67.611 | 18.222 | −3.806 | 0.5834 |
| classifier + geometry | 67.663 | **17.732** | −4.296 | 0.5983 |
| **classifier + geoMLP** | **67.757** | 19.547 | −2.480 | **0.6117** |

```
Spearman(R@50,   WPRD) = +0.914   p = 0.0005   significant
Spearman(mR@50,  WPRD) = -0.727   p = 0.0080   significant
Spearman(Pareto, WPRD) = -0.448   p = 0.1439   NOT significant
```
(permutation p, 2000 draws)

## Replication status

| | val raw-name (`p49`) | val VG150-only (`p53`) | **TEST (`p61`)** |
|---|---|---|---|
| Spearman(R@50, WPRD) | +0.741, p=0.0080 | +0.951, p=0.0005 | **+0.914, p=0.0005** |
| Spearman(mR@50, WPRD) | −0.650, p=0.0205 | −0.629, p=0.0255 | **−0.727, p=0.0080** |
| Spearman(Pareto, WPRD) | −0.371, ns | −0.112, ns | −0.448, ns |

**Items 4 and 5 hold: same sign, significant at p < 0.05, on held-out data.**
Both coefficients are *stronger* on test than on the raw-name validation split.

## The caveat that did NOT go away

The test split does nothing about the objection that actually matters. **The
twelve arms still cluster into ~3 families** (prior/random; the PURE α-sweep;
the geometry-containing arms), so the effective n is nearer 3 than 12 and the
permutation p-values remain optimistic. And **these are still twelve scoring
functions built from one checkpoint's cache, not twelve published SGG models.**

Held-out replication upgrades this from "validation-specific" to "a property of
this checkpoint's score family on unseen data". It does **not** upgrade it to a
statement about SGG. **The cross-model gate is untouched and still binding.**
