import json
import os
from pathlib import Path

def generate_vocab(vg_root, export_dir):
    vg_root = Path(vg_root)
    export_dir = Path(export_dir)
    vocab_dir = export_dir / "vocabulary"
    vocab_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Extracting vocab from {vg_root}...")
    
    # Đọc file dict gốc của VG150
    with open(vg_root / "VG-SGG-dicts-with-attri.json", 'r', encoding='utf-8') as f:
        source_dicts = json.load(f)

    # Xuất file objects.json (150 lớp)
    with open(vocab_dir / "objects.json", 'w') as f:
        json.dump({"idx_to_label": source_dicts["idx_to_label"]}, f, indent=2)
    
    # Xuất file predicates.json (50 lớp)
    with open(vocab_dir / "predicates.json", 'w') as f:
        json.dump({"idx_to_predicate": source_dicts["idx_to_predicate"]}, f, indent=2)

    print(f"[✓] Created: {vocab_dir / 'objects.json'}")
    print(f"[✓] Created: {vocab_dir / 'predicates.json'}")

if __name__ == "__main__":
    generate_vocab("datasets/vg150", "runs/vg150_hf_export")