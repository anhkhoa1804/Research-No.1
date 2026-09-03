"""
Converts the maelic/VG150-coco-format HF dataset (per-image COCO-style
objects/relations) into the VG-SGG.h5 / VG-SGG-dicts.json / image_data.json
trio expected by bknyaz/sgg's dataloaders.visual_genome.VG loader.

This exists because every repository that ships VG150 PredCls checkpoints
needs the original Xu/Zellers VG-SGG.h5, which is published only inside a
26 GB Yandex tarball. maelic/VG150-coco-format carries the same information
(confirmed below) in an openly downloadable per-image format instead.

Schema derived by reading dataloaders/visual_genome.py::load_graphs directly
(bknyaz/sgg), not assumed:

  labels            int32 (N,1)      canonical VG150 object index, 1..150
  boxes_1024        float32 (N,4)    (xc, yc, w, h), image scaled so
                                      max(width,height) == 1024
  img_to_first_box  int32 (N_img,)   inclusive range into labels/boxes_1024,
  img_to_last_box   int32 (N_img,)   -1 if the image has no objects
  img_to_first_rel  int32 (N_img,)   inclusive range into relationships/
  img_to_last_rel   int32 (N_img,)   predicates, -1 if the image has none
  relationships     int32 (M,2)      GLOBAL box indices [subject, object]
  predicates        int32 (M,1)      canonical VG150 predicate index, 1..50
  split             int32 (N_img,)   0 = train, 2 = test (val is carved from
                                      train by the loader itself and is not
                                      needed for a test-set PredCls eval)

VG-SGG-dicts.json: {"label_to_idx": {name: 1..150}, "predicate_to_idx":
{name: 1..50}} -- load_info() adds "__background__": 0 itself.

image_data.json: list of {"image_id": int, "width": int, "height": int},
in the SAME row order as the h5 arrays (position-aligned, load-bearing).

Reversible: every field here is a deterministic function of the source
parquet row; re-running on the same input parquet reproduces byte-identical
output. Nothing here mutates or overwrites the source parquet or the raw
VG image corpus.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq


def load_categories(categories_path: Path):
    cats = json.loads(categories_path.read_text())
    obj_cats = {c["id"]: c["name"] for c in cats["categories"]}
    rel_cats = {c["id"]: c["name"] for c in cats["rel_categories"]}

    assert len(obj_cats) == 150, len(obj_cats)
    assert len(rel_cats) == 50, len(rel_cats)
    assert set(obj_cats.keys()) == set(range(1, 151))
    assert set(rel_cats.keys()) == set(range(1, 51))

    # Load-bearing assumption, verified here rather than assumed: the HF
    # dataset's category "id" already equals the canonical Xu/Zellers
    # alphabetical 1..150 / 1..50 index (bknyaz's VG-SGG-dicts.json is
    # exactly `sorted(name) -> 1-based rank`). If this ever fails, the
    # checkpoint's predicate head would be silently misaligned.
    sorted_obj_names = sorted(obj_cats.values())
    for rank, name in enumerate(sorted_obj_names, start=1):
        matching_id = [i for i, n in obj_cats.items() if n == name][0]
        assert matching_id == rank, (
            f"object category id/alphabetical-rank mismatch: "
            f"{name!r} has id {matching_id}, alphabetical rank {rank}"
        )
    sorted_rel_names = sorted(rel_cats.values())
    for rank, name in enumerate(sorted_rel_names, start=1):
        matching_id = [i for i, n in rel_cats.items() if n == name][0]
        assert matching_id == rank, (
            f"predicate id/alphabetical-rank mismatch: "
            f"{name!r} has id {matching_id}, alphabetical rank {rank}"
        )

    return obj_cats, rel_cats


def convert(annotations_parquet: Path, categories_path: Path, out_dir: Path,
            split_name: str, split_value: int, box_scale: int = 1024):
    assert split_name in ("train", "test")
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_cats, rel_cats = load_categories(categories_path)

    table = pq.read_table(
        annotations_parquet,
        columns=["image_id", "width", "height", "file_name", "objects", "relations"],
    )
    rows = table.to_pylist()
    print(f"[{split_name}] {len(rows)} images read from {annotations_parquet}")

    all_labels, all_boxes = [], []
    img_to_first_box, img_to_last_box = [], []
    all_rel_subj_obj, all_rel_pred = [], []
    img_to_first_rel, img_to_last_rel = [], []
    image_data = []
    n_relations_total = 0
    n_dropped_zero_area = 0

    for row in rows:
        w, h = row["width"], row["height"]
        s = float(box_scale) / max(w, h)

        objs = row["objects"]
        # Load-bearing assumption, checked per-row: objects[i]["id"] == i
        # (sequential local 0-based index), which is exactly what
        # relations[*].subject_id / object_id reference.
        for local_idx, o in enumerate(objs):
            assert o["id"] == local_idx, (
                f"image {row['image_id']}: objects[{local_idx}]['id']={o['id']}, "
                "expected sequential local index -- converter assumption violated"
            )

        first_box = len(all_labels)
        kept_local_to_global = {}
        for local_idx, o in enumerate(objs):
            x, y, bw, bh = o["bbox"]
            if bw <= 0 or bh <= 0:
                n_dropped_zero_area += 1
                continue
            kept_local_to_global[local_idx] = len(all_labels)
            all_labels.append(o["category_id"])
            xc = (x + bw / 2.0) * s
            yc = (y + bh / 2.0) * s
            all_boxes.append([xc, yc, bw * s, bh * s])
        last_box = len(all_labels) - 1

        if last_box < first_box:
            img_to_first_box.append(-1)
            img_to_last_box.append(-1)
        else:
            img_to_first_box.append(first_box)
            img_to_last_box.append(last_box)

        first_rel = len(all_rel_pred)
        for r in row["relations"]:
            sub_local, obj_local = r["subject_id"], r["object_id"]
            if sub_local not in kept_local_to_global or obj_local not in kept_local_to_global:
                continue  # referenced a zero-area object we dropped
            all_rel_subj_obj.append(
                [kept_local_to_global[sub_local], kept_local_to_global[obj_local]]
            )
            all_rel_pred.append(r["predicate_id"])
        last_rel = len(all_rel_pred) - 1
        n_relations_total += max(0, last_rel - first_rel + 1)

        if last_rel < first_rel:
            img_to_first_rel.append(-1)
            img_to_last_rel.append(-1)
        else:
            img_to_first_rel.append(first_rel)
            img_to_last_rel.append(last_rel)

        image_data.append(
            {"image_id": row["image_id"], "width": w, "height": h,
             "file_name": row["file_name"]}
        )

    n_img = len(rows)
    split_arr = np.full((n_img,), split_value, dtype=np.int32)

    h5_path = out_dir / f"VG-SGG-{split_name}.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("labels", data=np.asarray(all_labels, dtype=np.int32).reshape(-1, 1))
        f.create_dataset(f"boxes_{box_scale}", data=np.asarray(all_boxes, dtype=np.float32))
        f.create_dataset("img_to_first_box", data=np.asarray(img_to_first_box, dtype=np.int32))
        f.create_dataset("img_to_last_box", data=np.asarray(img_to_last_box, dtype=np.int32))
        f.create_dataset("img_to_first_rel", data=np.asarray(img_to_first_rel, dtype=np.int32))
        f.create_dataset("img_to_last_rel", data=np.asarray(img_to_last_rel, dtype=np.int32))
        f.create_dataset("relationships", data=np.asarray(all_rel_subj_obj, dtype=np.int32).reshape(-1, 2))
        f.create_dataset("predicates", data=np.asarray(all_rel_pred, dtype=np.int32).reshape(-1, 1))
        f.create_dataset("split", data=split_arr)

    dicts_path = out_dir / "VG-SGG-dicts.json"
    if not dicts_path.exists():
        label_to_idx = {name: idx for idx, name in obj_cats.items()}
        predicate_to_idx = {name: idx for idx, name in rel_cats.items()}
        dicts_path.write_text(json.dumps(
            {"label_to_idx": label_to_idx, "predicate_to_idx": predicate_to_idx}, indent=2
        ))

    image_data_path = out_dir / f"image_data-{split_name}.json"
    image_data_path.write_text(json.dumps(image_data))

    print(f"[{split_name}] wrote {h5_path}")
    print(f"[{split_name}] images={n_img} objects={len(all_labels)} "
          f"(dropped {n_dropped_zero_area} zero-area) relations={n_relations_total}")

    return {
        "n_images": n_img,
        "n_objects": len(all_labels),
        "n_relations": n_relations_total,
        "n_dropped_zero_area": n_dropped_zero_area,
        "h5_path": str(h5_path),
        "dicts_path": str(dicts_path),
        "image_data_path": str(image_data_path),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations-parquet", type=Path, required=True)
    ap.add_argument("--categories-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split-name", choices=["train", "test"], required=True)
    ap.add_argument("--split-value", type=int, required=True, help="0=train, 2=test")
    args = ap.parse_args()

    stats = convert(args.annotations_parquet, args.categories_json, args.out_dir,
                     args.split_name, args.split_value)
    print(json.dumps(stats, indent=2))
