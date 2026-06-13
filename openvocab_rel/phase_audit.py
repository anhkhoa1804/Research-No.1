from __future__ import annotations

from typing import Any, Dict


def pure_phase_audit(cfg: Any) -> Dict[str, Dict[str, Any]]:
    """Report whether the five PURE extension phases are enabled/configured."""
    return {
        "sgcls_bridge": {
            "implemented": True,
            "enabled": bool(getattr(cfg, "eval_sgg_use_clip_obj_classifier", True) or getattr(cfg, "eval_sgg_sgcls_oracle_labels", False)),
            "evidence": "SGCls evaluation uses GT boxes with predicted or oracle object labels.",
        },
        "sgdet_bridge": {
            "implemented": True,
            "enabled": bool(getattr(cfg, "eval_sgg_grounding_dino_enabled", True)),
            "evidence": "SGDet evaluation can use Grounding DINO proposals plus PURE predicate ranking.",
        },
        "one_stage_facade": {
            "implemented": True,
            "enabled": bool(getattr(cfg, "one_stage_facade_enabled", False)),
            "evidence": "RelationalModel.forward_one_stage_facade standardizes detector proposals -> relation features.",
        },
        "open_vocab_predicates": {
            "implemented": True,
            "enabled": bool(getattr(cfg, "text_conditioned_projection_enabled", False) or getattr(cfg, "open_vocab_predicate_primary", False)),
            "evidence": "predicate_scores can use text predicate embeddings as primary or ensemble logits.",
        },
        "retrieval_index": {
            "implemented": True,
            "enabled": bool(getattr(cfg, "retrieval_index_enabled", False)),
            "evidence": "TripletRetrievalIndex stores predicted triplets and searches by text/query embeddings.",
        },
    }


def summarize_pure_phase_audit(cfg: Any) -> str:
    audit = pure_phase_audit(cfg)
    rows = []
    for name, info in audit.items():
        status = "enabled" if bool(info.get("enabled", False)) else "available"
        rows.append(f"{name}: {status} — {info.get('evidence', '')}")
    return "\n".join(rows)