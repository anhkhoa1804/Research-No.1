"""
Monkeypatch for dataloaders.visual_genome.load_image_filenames.

The upstream function hardcodes `assert len(fns) == 108073` (the exact size
of the original raw VG corpus) and silently drops any image_data.json entry
whose file does not exist on disk -- both wrong for our converted, test-only
image_data.json. This replacement preserves positional alignment (required:
h5 arrays are indexed by position against this list) and fails loudly
instead of silently dropping a row.
"""
import json
import os


def load_image_filenames_patched(image_file, image_dir):
    with open(image_file, "r") as f:
        im_data = json.load(f)

    fns = []
    for img in im_data:
        basename = "{}.jpg".format(img["image_id"])
        full = os.path.join(image_dir, basename)
        if not os.path.exists(full):
            raise FileNotFoundError(
                f"image referenced by image_data.json not found: {full}"
            )
        fns.append(basename)
    return fns


def apply():
    import dataloaders.visual_genome as vgmod
    vgmod.load_image_filenames = load_image_filenames_patched
