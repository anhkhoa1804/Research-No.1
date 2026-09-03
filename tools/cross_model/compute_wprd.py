"""
Computes WPRD on the extracted cross-model pairs, using the exact `auc()`
function from tools/within_pair_discrimination.py (imported, not
reimplemented) so tie-handling and the AUC definition are byte-identical to
every WPRD number elsewhere in this project.

Also runs the prior-control check: with score held CONSTANT within every
(subject, object) group (here, the empirical within-group predicate
frequency), WPRD must read exactly 0.5000 by construction (see that file's
docstring for the proof) -- this is a check on THIS script's own grouping
logic, not on the external model.
"""
import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path("/home/leanhkhoa150204/Research-No.1")


def _load_auc():
    spec = importlib.util.spec_from_file_location(
        "within_pair_discrimination", str(REPO_ROOT / "tools" / "within_pair_discrimination.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.auc


AUC = _load_auc()


class SimpleGroups:
    """Duck-types the attributes tools/within_pair_discrimination.py::wprd() needs
    (Gs.G, Gs.classes_of, Gs.rows_of, Gs.y), built directly from subject/object
    category name lists -- no dependency on this project's own Mech cache format.
    """

    def __init__(self, subj_label: List[str], obj_label: List[str], y: torch.Tensor):
        keys = [f"{s}||{o}" for s, o in zip(subj_label, obj_label)]
        uniq = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        self.G = len(uniq)
        pair_id = [uniq[k] for k in keys]
        self.y = y
        self.rows_of: Dict[int, List[int]] = defaultdict(list)
        for r, p in enumerate(pair_id):
            self.rows_of[p].append(r)
        self.classes_of: Dict[int, set] = defaultdict(set)
        for p, rows in self.rows_of.items():
            self.classes_of[p] = set(y[torch.tensor(rows)].tolist())


def wprd_generic(Gs: SimpleGroups, score: torch.Tensor, cap: int = 64, seed: int = 0):
    """Verbatim port of tools/within_pair_discrimination.py::wprd()'s algorithm,
    against SimpleGroups instead of the Mech-coupled Groups class."""
    gen = torch.Generator().manual_seed(seed)
    vals, wts = [], []
    n_cells = 0
    for p in range(Gs.G):
        cls = sorted(Gs.classes_of[p])
        if len(cls) < 2:
            continue
        rows = torch.tensor(Gs.rows_of[p])
        yy = Gs.y[rows]
        byc = {c: rows[yy == c] for c in cls}
        for ai in range(len(cls)):
            for bi in range(ai + 1, len(cls)):
                a, b = cls[ai], cls[bi]
                ra, rb = byc[a], byc[b]
                if len(ra) > cap:
                    ra = ra[torch.randperm(len(ra), generator=gen)[:cap]]
                if len(rb) > cap:
                    rb = rb[torch.randperm(len(rb), generator=gen)[:cap]]
                sa = score[ra, a] - score[ra, b]
                sb = score[rb, a] - score[rb, b]
                v = AUC(sa.double(), sb.double())
                w = len(ra) * len(rb)
                vals.append(v)
                wts.append(w)
                n_cells += 1
    v = torch.tensor(vals, dtype=torch.float64)
    w = torch.tensor(wts, dtype=torch.float64)
    return {
        "wprd_macro": float(v.mean()) if len(vals) else float("nan"),
        "wprd_weighted": float((v * w).sum() / w.sum()) if len(vals) else float("nan"),
        "n_cells": n_cells,
        "n_comparisons": int(w.sum()) if len(vals) else 0,
    }


def empirical_group_prior(Gs: SimpleGroups, n_rows: int, n_classes: int = 50) -> torch.Tensor:
    """Per-row score = empirical within-group predicate frequency -- constant
    across rows of the same group by construction. WPRD on this must read
    exactly 0.5000 (a proof, not a measurement -- see module docstring)."""
    out = torch.zeros(n_rows, n_classes)
    for p, rows in Gs.rows_of.items():
        rows_t = torch.tensor(rows)
        counts = torch.zeros(n_classes)
        for r in rows:
            counts[int(Gs.y[r])] += 1
        freq = counts / counts.sum()
        out[rows_t] = freq
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help=".pt from extract_wprd_pairs.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = torch.load(args.pairs, map_location="cpu", weights_only=False)
    model_term = d["model_term"]
    gt_y = d["gt_y"]
    subj_label = d["subj_label"]
    obj_label = d["obj_label"]
    n = model_term.shape[0]
    print(f"loaded {n} GT-aligned pairs")

    Gs = SimpleGroups(subj_label, obj_label, gt_y)
    n_singleton = sum(1 for rows in Gs.rows_of.values() if len(rows) < 2)
    n_decidable_groups = sum(1 for p in range(Gs.G) if len(Gs.classes_of[p]) >= 2)
    print(f"n_groups={Gs.G} n_singleton_groups={n_singleton} "
          f"n_decidable_groups={n_decidable_groups}")

    prior = empirical_group_prior(Gs, n, 50)
    prior_result = wprd_generic(Gs, prior)
    print(f"PRIOR CONTROL (must be exactly 0.5000): "
          f"wprd_macro={prior_result['wprd_macro']:.6f} "
          f"wprd_weighted={prior_result['wprd_weighted']:.6f} "
          f"n_cells={prior_result['n_cells']}")

    model_result = wprd_generic(Gs, model_term)
    print(f"MODEL (IMP+ text/rel_fc head): "
          f"wprd_macro={model_result['wprd_macro']:.6f} "
          f"wprd_weighted={model_result['wprd_weighted']:.6f} "
          f"n_cells={model_result['n_cells']} "
          f"n_comparisons={model_result['n_comparisons']}")

    out = {
        "n_gt_rows": n,
        "n_groups": Gs.G,
        "n_singleton_groups": n_singleton,
        "n_decidable_groups": n_decidable_groups,
        "prior_control": prior_result,
        "model": model_result,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
