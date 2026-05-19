import json
import os
from typing import Dict, List, Optional

import torch


SYMMETRIC_RELATION_BLACKLIST = {
    "near",
    "next to",
    "beside",
    "overlap",
    "overlapping",
    "on either side of",
    "looking at each other",
    "adjacent to",
    "around",
    "surrounding",
}


def is_symmetric_predicate(pred: str) -> bool:
    return str(pred).strip().lower() in SYMMETRIC_RELATION_BLACKLIST


def pred_prompt(pred: str, direction: str = "s2o") -> str:
    if pred is None:
        pred = ""
    pred = str(pred).strip().lower()
    if pred == "":
        pred = "relation"
    if direction == "o2s":
        return f"predicate: {pred} (object->subject)"
    return f"predicate: {pred} (subject->object)"


def pred_prompt_ensemble(pred: str, direction: str = "s2o") -> List[str]:
    p = str(pred).strip().lower()
    if p == "":
        p = "relation"
    d = "object->subject" if direction == "o2s" else "subject->object"
    return [
        f"predicate: {p} ({d})",
        f"relationship predicate: {p} ({d})",
        f"relation: {p} ({d})",
    ]


def pred_prompt_roles(pred: str, direction: str = "s2o") -> str:
    p = str(pred).strip().lower()
    if p == "":
        p = "relation"
    if direction == "o2s":
        return f"predicate: {p} | roles: [OBJ]->[SUBJ]"
    return f"predicate: {p} | roles: [SUBJ]->[OBJ]"


def counterfactual_prompt_variants(pred: str, s_name: str, o_name: str, direction: str = "s2o") -> Dict[str, str]:
    p = str(pred).strip().lower()
    if p == "":
        p = "relation"
    s_name = str(s_name).strip().lower() if s_name is not None else "object"
    o_name = str(o_name).strip().lower() if o_name is not None else "object"

    d = "object->subject" if direction == "o2s" else "subject->object"

    base = pred_prompt(p, direction=direction)
    placeholder = f"relationship between [SUBJ] and [OBJ]: {p} ({d})"
    with_names = f"relationship between {s_name} and {o_name}: {p} ({d})"
    with_names_swapped = f"relationship between {o_name} and {s_name}: {p} ({d})"

    return {
        "base": base,
        "placeholder": placeholder,
        "with_names": with_names,
        "with_names_swapped": with_names_swapped,
    }


class PredicateCanonicalizer:
    def __init__(
        self,
        enabled: bool = False,
        model_name: str = "google/flan-t5-small",
        device: str = "cpu",
        cache_path: str = "",
    ):
        self.enabled = bool(enabled)
        self.model_name = str(model_name)
        self.device = torch.device(device)
        self.cache_path = str(cache_path)
        self._tok = None
        self._mdl = None
        self._para_cache: Dict[str, List[str]] = {}

        if self.cache_path != "" and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    self._para_cache = {str(k): [str(x) for x in v] for k, v in obj.items() if isinstance(v, list)}
            except Exception:
                self._para_cache = {}

    def save_cache(self) -> None:
        if self.cache_path == "":
            return
        try:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._para_cache, f, ensure_ascii=False)
        except Exception:
            pass

    def _lazy_init(self) -> None:
        if self._tok is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "PredicateCanonicalizer requires `transformers`. Install it or disable canonicalization (set --canon_enabled false)."
            ) from e

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._mdl = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self._mdl.eval()

    @torch.no_grad()
    def canonicalize(self, pred: str) -> str:
        pred = str(pred).strip().lower()
        if (not self.enabled) or pred == "":
            return pred
        self._lazy_init()
        prompt = (
            "Canonicalize this relation predicate to a short verb phrase (1-3 words), "
            "lowercase, no punctuation. Input: "
            + pred
        )
        inputs = self._tok([prompt], return_tensors="pt", padding=True).to(self.device)
        out = self._mdl.generate(**inputs, max_new_tokens=8)
        s = self._tok.decode(out[0], skip_special_tokens=True)
        s = str(s).strip().lower()
        return s if s != "" else pred

    @torch.no_grad()
    def generate_paraphrases(self, pred: str, n: int = 5) -> List[str]:
        pred = str(pred).strip().lower()
        if (not self.enabled) or pred == "":
            return []

        if pred in self._para_cache and len(self._para_cache[pred]) > 0:
            return [str(x) for x in self._para_cache[pred]][: int(n)]

        self._lazy_init()
        prompt = (
            "Generate "
            + str(int(n))
            + " paraphrases of this relation predicate as short verb phrases (1-4 words), lowercase, no punctuation: "
            + pred
        )
        inputs = self._tok([prompt], return_tensors="pt", padding=True).to(self.device)
        out = self._mdl.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=max(1, int(n)),
            num_return_sequences=max(1, int(n)),
        )
        outs = []
        for i in range(out.shape[0]):
            s = self._tok.decode(out[i], skip_special_tokens=True)
            s = str(s).strip().lower()
            if s != "":
                outs.append(s)

        uniq = []
        seen = set()
        for s in outs:
            if s in seen:
                continue
            seen.add(s)
            uniq.append(s)

        self._para_cache[pred] = uniq
        return uniq[: int(n)]


def build_predicate_prompt_candidates(
    pred: str,
    direction: str,
    canon: Optional[PredicateCanonicalizer],
    max_paraphrases: int,
) -> List[str]:
    p = str(pred).strip().lower()
    if p == "":
        p = "relation"

    out: List[str] = []
    out.append(pred_prompt(p, direction=direction))
    out.extend(pred_prompt_ensemble(p, direction=direction))
    out.append(pred_prompt_roles(p, direction=direction))

    if canon is not None and bool(getattr(canon, "enabled", False)) and int(max_paraphrases) > 0:
        paras = canon.generate_paraphrases(p, n=int(max_paraphrases))
        for pp in paras:
            if str(pp).strip() == "":
                continue
            out.append(pred_prompt(pp, direction=direction))
            out.append(pred_prompt_roles(pp, direction=direction))

    uniq: List[str] = []
    seen = set()
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)

    return uniq
