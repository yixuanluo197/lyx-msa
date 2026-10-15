import importlib
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_official_model_modules_import_with_declared_pyg_version():
    for module_name in ("HypergraphConv", "model_GCN", "model_hyper", "model", "model_cm"):
        importlib.import_module(module_name)
