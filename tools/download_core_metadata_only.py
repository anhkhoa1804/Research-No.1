from __future__ import annotations

import argparse
import importlib
import inspect
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, List


def extract_folder_id(url_or_id: str) -> str:
    text = str(url_or_id).strip()
    if "/folders/" in text:
        return text.split("/folders/", 1)[1].split("?", 1)[0].split("/", 1)[0]
    return text


def item_name(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("name") or item.get("filename") or item.get("title") or "")
    return str(getattr(item, "name", getattr(item, "filename", getattr(item, "title", ""))))


def is_folder(item: Any) -> bool:
    if isinstance(item, dict):
        return item.get("type") == "folder" or bool(item.get("children") or item.get("folders"))
    return getattr(item, "type", None) == "folder" or bool(getattr(item, "children", None) or getattr(item, "folders", None))


def child_nodes(item: Any) -> Iterable[Any]:
    if item is None:
        return []
    if isinstance(item, dict):
        children = []
        for key in ("children", "files", "folders"):
            value = item.get(key)
            if isinstance(value, (list, tuple)):
                children.extend(value)
        return children
    children = []
    for key in ("children", "files", "folders"):
        value = getattr(item, key, None)
        if isinstance(value, (list, tuple)):
            children.extend(value)
    return children


def iter_file_objects(node: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if node is None:
        return
    if isinstance(node, (list, tuple, set)):
        for item in node:
            yield from iter_file_objects(item, prefix)
        return
    name = item_name(node)
    current_path = f"{prefix}/{name}".strip("/") if name else prefix
    children = list(child_nodes(node))
    if children or is_folder(node):
        for child in children:
            yield from iter_file_objects(child, current_path)
        return
    if item_id(node) or (isinstance(node, dict) and node.get("url")):
        yield current_path, node

def item_id(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("id") or "")
    return str(getattr(item, "id", ""))


def item_path(item: Any) -> str:
    parts = []
    current = item
    while current is not None:
        name = item_name(current)
        if name:
            parts.append(name)
        current = getattr(current, "parent", None) if not isinstance(current, dict) else current.get("parent")
    if parts:
        return "/".join(reversed(parts))
    return item_name(item)


def call_parse_google_drive_file(folder_id: str) -> Any:
    parse_module = importlib.import_module("gdown.download_folder")
    parser = getattr(parse_module, "_parse_google_drive_file", None)
    if parser is None:
        parser = getattr(parse_module, "parse_google_drive_file", None)
    if parser is None:
        raise RuntimeError("This gdown version does not expose a folder parser.")
    signature = inspect.signature(parser)
    kwargs = {}
    if "quiet" in signature.parameters:
        kwargs["quiet"] = False
    if "remaining_ok" in signature.parameters:
        kwargs["remaining_ok"] = True
    try:
        return parser(folder_id, **kwargs)
    except TypeError:
        return parser(f"https://drive.google.com/drive/folders/{folder_id}", **kwargs)


def download_file(file_id: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "gdown", file_id, "-O", str(output)]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download only CORE metadata.json files from the public Google Drive folder.")
    parser.add_argument("--folder-url", default="https://drive.google.com/drive/folders/11rAVJgxZ557XPf4JHyQi7rJ7Hmw7fMHR")
    parser.add_argument("--out-root", default="datasets/core/dataset")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    folder_id = extract_folder_id(args.folder_url)
    out_root = Path(args.out_root)
    tree = call_parse_google_drive_file(folder_id)
    files = list(iter_file_objects(tree))
    metadata = [(rel, item) for rel, item in files if item_name(item) == "metadata.json"]
    if not metadata:
        raise SystemExit(f"No metadata.json files discovered in Drive folder {folder_id}; discovered files={len(files)}")

    print(f"Discovered {len(metadata)} metadata.json files")
    for rel, item in metadata:
        parts = rel.split("/")
        if parts and parts[0] not in {"v1", "v2", "dataset"}:
            parts = parts[1:]
        rel = "/".join(parts)
        if rel.startswith("dataset/"):
            rel = rel[len("dataset/"):]
        if not rel.endswith("metadata.json"):
            rel = f"{rel}/metadata.json"
        target = out_root / rel
        print(f"{item_id(item)} -> {target}")
        if not args.dry_run:
            download_file(item_id(item), target)


if __name__ == "__main__":
    main()