"""Import-graph smoke test. No external data, no GPU, no network access."""
import importlib

import pytest


def test_import_top_level_package():
    importlib.import_module("openvocab_rel")


@pytest.mark.parametrize(
    "module_name",
    [
        "openvocab_rel.config",
        "openvocab_rel.geometry",
        "openvocab_rel.lattice",
        "openvocab_rel.losses",
        "openvocab_rel.prompts",
        "openvocab_rel.predicate_metadata",
        "openvocab_rel.retrieval",
        "openvocab_rel.text_cache",
        "openvocab_rel.phase_audit",
        "openvocab_rel.ddp_utils",
        "openvocab_rel.fp8_utils",
        "openvocab_rel.models",
        "openvocab_rel.models.relational_model",
        "openvocab_rel.datasets",
        "openvocab_rel.datasets.vg150_loader",
    ],
)
def test_import_submodule(module_name):
    importlib.import_module(module_name)


def test_import_clip_dependent_modules():
    """clip_utils.py, evals.py, train.py import `transformers` at module
    level. Importing them never downloads a model (that only happens when
    something later calls e.g. CLIPModel.from_pretrained) but does require
    transformers to be installed -- skip cleanly if it isn't, so the rest of
    the suite stays runnable in a minimal environment.
    """
    pytest.importorskip("transformers")
    importlib.import_module("openvocab_rel.clip_utils")
    importlib.import_module("openvocab_rel.evals")
    importlib.import_module("openvocab_rel.train")
