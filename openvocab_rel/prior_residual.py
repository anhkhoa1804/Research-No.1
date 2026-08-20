"""Prior-residual predicate scoring (architecture candidate A1).

MOTIVATION
----------
A pair-conditioned co-occurrence table, built only from the training split,
already reaches R@50 = 66.59 % / mR@50 = 22.30 % on VG150 validation with no
model at all (``tools/frequency_prior_baseline.py``). The historical
model+prior system reached 67.09 / 22.64 -- so the learned model contributes
roughly +0.5 R@50 and +0.3 mR@50 over a lookup table.

The reason is structural, not a tuning failure: a 50-way classifier trained
with plain cross-entropy is *rewarded* for reproducing the label marginal
P(p | s, o), which is exactly what the table already stores. Its capacity
goes into re-deriving statistics it is later handed for free.

This module supplies that information explicitly during training, so the
gradient the model receives is proportional to what the prior gets WRONG.

WHAT THE MODEL IS OPTIMISED TO PREDICT
--------------------------------------
With ``z = alpha * log P(p | s, o) + f_theta(x)`` and ordinary cross-entropy,
the optimum satisfies ``softmax(z) = P(p | x, s, o)``, hence

    f_theta*(x) = log P(p | x, s, o) - alpha * log P(p | s, o) + const

At ``alpha = 1`` that is exactly the pointwise log-likelihood ratio -- the
information the image adds *over* co-occurrence. This is why the default
alpha here is 1.0 and NOT the historical evaluation-time 3.75: at 3.75 the
prior is over-counted and f_theta is forced to spend capacity cancelling
2.75 * log P(p | s, o) before it can express anything visual.

WHAT THIS DOES *NOT* GUARANTEE
------------------------------
It removes the *incentive* to relearn the prior; it does not make it
impossible. ``f_theta`` still sees subject/object appearance and can in
principle re-derive a label-conditioned term. Nothing here proves the model
uses vision -- only the visual ablation (``visual_ablation_mode``) can, and
it is a hard gate, not a formality.

Also note: stop-gradient on the prior is a mathematical no-op while the prior
is a fixed lookup table with no parameters. It is implemented and configurable
because it stops being a no-op the moment the prior is composed with a learned
calibration gate (``adaptive_calibration_enabled``), and being explicit costs
nothing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


class _PriorCfgShim:
    """Minimal cfg surface for evals._load_frequency_bias.

    The prior loader is reused verbatim rather than reimplemented, so the
    training-time prior and the evaluation-time prior are parsed, remapped to
    the model's predicate ordering, and backed off identically. Duplicating
    that logic is how the two silently diverge.
    """

    def __init__(self, path: str, smoothing: float = 1.0):
        self.freq_bias_enabled = True
        self.freq_bias_path = str(path)
        self.freq_bias_smoothing = float(smoothing)


class PairPriorTable:
    """Pair-conditioned log P(predicate | subject_label, object_label)."""

    def __init__(self, table: Dict[str, Any], path: str):
        self._table = table
        self.path = str(path)
        self.num_source_predicates = int(table.get("num_source_predicates", 0))

    @classmethod
    def load(
        cls,
        path: str,
        pred_vocab: Sequence[str],
        device: torch.device,
        smoothing: float = 1.0,
    ) -> Optional["PairPriorTable"]:
        """Load via the evaluator's own loader. Returns None if unusable.

        ``_load_frequency_bias`` fails silently by design (six separate
        ``return None`` paths, see docs/known_issues.md). Callers MUST treat
        None as fatal when the prior was requested -- silently training
        without it would produce a run that looks like A1 but is A0.
        """
        from .evals import _load_frequency_bias

        table = _load_frequency_bias(_PriorCfgShim(path, smoothing), list(pred_vocab), device)
        if table is None:
            return None
        return cls(table, path)

    def logits_for_pairs(
        self,
        pair_names: Sequence[Tuple[str, str]],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """[N, P] log-prior rows for (subject_label, object_label) pairs.

        Reuses ``_frequency_bias_for_pairs`` by synthesising a flat label list,
        so the pair -> subject -> object -> global backoff chain is byte-for-byte
        the same one evaluation uses.
        """
        from .evals import _frequency_bias_for_pairs

        if len(pair_names) == 0:
            return None
        obj_labels: List[str] = []
        pair_list: List[Tuple[int, int]] = []
        for subj, obj in pair_names:
            i = len(obj_labels)
            obj_labels.append(str(subj))
            j = len(obj_labels)
            obj_labels.append(str(obj))
            pair_list.append((i, j))
        return _frequency_bias_for_pairs(self._table, pair_list, obj_labels, device)


def compose_prior_residual(
    model_logits: torch.Tensor,
    prior_logits: Optional[torch.Tensor],
    alpha: float = 1.0,
    stopgrad: bool = True,
    center_prior: bool = True,
) -> torch.Tensor:
    """z = alpha * log_prior + f_theta(x)   (candidate A1).

    ``center_prior`` subtracts the per-row mean of the prior. This is a shift
    along the all-ones direction, to which softmax is invariant, so it changes
    neither the loss nor any gradient -- it only keeps the logits numerically
    centred, which matters because raw log-probabilities span roughly -9..0
    and would otherwise push the whole logit vector far negative.
    """
    if prior_logits is None or float(alpha) == 0.0:
        return model_logits
    if tuple(prior_logits.shape) != tuple(model_logits.shape):
        raise ValueError(
            f"prior/model logit shape mismatch: {tuple(prior_logits.shape)} vs "
            f"{tuple(model_logits.shape)}. Refusing to broadcast silently."
        )
    prior = prior_logits.to(device=model_logits.device, dtype=model_logits.dtype)
    if stopgrad:
        prior = prior.detach()
    if center_prior:
        prior = prior - prior.mean(dim=-1, keepdim=True)
    return model_logits + (float(alpha) * prior)


def residual_diagnostics(
    model_logits: torch.Tensor,
    prior_logits: Optional[torch.Tensor],
    alpha: float,
    targets: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Magnitudes needed to tell a real residual from a numerically dead one.

    Acceptance criterion 3 of the phase brief is that the residual must not be
    negligible against the prior term. ``residual_to_prior_ratio`` is that
    number: if it sits near zero the model is a pass-through and A1 has
    degenerated into P0.
    """
    out: Dict[str, float] = {}
    with torch.no_grad():
        m = model_logits.float()
        out["residual_abs_mean"] = float(m.abs().mean())
        out["residual_std"] = float(m.std()) if m.numel() > 1 else 0.0
        if prior_logits is not None:
            p = (float(alpha) * prior_logits.float())
            p = p - p.mean(dim=-1, keepdim=True)
            out["prior_abs_mean"] = float(p.abs().mean())
            denom = out["prior_abs_mean"]
            out["residual_to_prior_ratio"] = float(out["residual_abs_mean"] / denom) if denom > 1e-8 else 0.0
            total = m + p
            out["prior_argmax_agrees_with_total"] = float(
                (p.argmax(dim=-1) == total.argmax(dim=-1)).float().mean()
            )
            if targets is not None and targets.numel() > 0:
                valid = (targets >= 0) & (targets < m.shape[-1])
                if bool(valid.any()):
                    t = targets[valid]
                    out["prior_top1_acc"] = float((p[valid].argmax(dim=-1) == t).float().mean())
                    out["total_top1_acc"] = float((total[valid].argmax(dim=-1) == t).float().mean())
                    out["model_only_top1_acc"] = float((m[valid].argmax(dim=-1) == t).float().mean())
    return out


def apply_visual_ablation(tokens: torch.Tensor, mode: str, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Hard scientific gate: does the model actually use the image?

    Applied to the dense CLIP patch tokens, i.e. the only route by which image
    content enters the relation encoder. Object boxes and geometry are left
    intact, so what is ablated is *appearance*, not the pair structure.

        none     unmodified
        zero     all visual evidence removed
        shuffle  tokens permuted across the batch, so each pair is scored
                 against a real but WRONG image -- controls for "the model
                 just needs some plausible activations"

    Expected pattern if the model uses vision:  none > shuffle >= zero.
    If none ~= shuffle ~= zero, the architecture failed regardless of metric.
    """
    m = str(mode).strip().lower()
    if m in ("", "none", "off"):
        return tokens
    if m == "zero":
        return torch.zeros_like(tokens)
    if m == "shuffle":
        n = int(tokens.shape[0])
        if n <= 1:
            return tokens
        perm = torch.randperm(n, device=tokens.device, generator=generator)
        # guarantee a derangement for n == 2 so the ablation is never a no-op
        if n == 2 and bool((perm == torch.arange(2, device=tokens.device)).all()):
            perm = torch.tensor([1, 0], device=tokens.device)
        return tokens.index_select(0, perm)
    raise ValueError(f"unknown visual_ablation_mode: {mode!r} (want none|zero|shuffle)")
