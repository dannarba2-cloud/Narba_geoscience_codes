"""Every registry method runs through the controller on its demo dataset, and projects round-trip."""

import numpy as np
import pytest

from core.registry import REGISTRY, numbered_count
from desktop.controller import ProjectController


@pytest.fixture(scope="module")
def controller():
    c = ProjectController()
    c.load_demo_library()
    return c


def _demo_inputs(method):
    inputs = {}
    for spec in method.inputs:
        if spec.optional:
            inputs[spec.name] = None
        elif isinstance(spec.demo, tuple):
            inputs[spec.name] = list(spec.demo)
        else:
            inputs[spec.name] = spec.demo
    return inputs


def test_catalogue_covers_all_125_numbered_methods():
    assert numbered_count() == 125
    keys = [m.key for m in REGISTRY]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("method", REGISTRY, ids=[m.key for m in REGISTRY])
def test_method_runs_on_demo_data(controller, method):
    before = {n: controller.session.get_object(n) for n in controller.session._all_names()}
    result = controller.run_method(method.key, _demo_inputs(method), {})
    assert result["outputs"] or result["report"] is not None
    for kind, name in result["outputs"]:
        obj = controller.session.get_object(name)
        if kind == "grid":
            assert obj.values.ndim == 2 and np.isfinite(obj.values).any()
        elif kind == "section":
            assert np.isfinite(obj.data).all()
    for name, obj in before.items():  # inputs untouched
        assert controller.session.get_object(name) is obj


def test_project_roundtrip_with_all_data_types(controller, tmp_path):
    path = tmp_path / "library.gflp"
    controller.save_project(path)
    restored = ProjectController()
    restored.load_project(path)
    for store in ("layers", "profiles", "sections", "tables", "images"):
        assert list(getattr(restored.session, store)) == list(getattr(controller.session, store))
    assert len(restored.session.reports) == len(controller.session.reports)
