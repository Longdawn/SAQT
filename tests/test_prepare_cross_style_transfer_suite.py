import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "prepare_cross_style_transfer_suite.py"
SPEC = importlib.util.spec_from_file_location("prepare_cross_style_transfer_suite", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def records(count):
    return [{"id": f"sample_{index:03d}"} for index in range(count)]


def test_nested_train_manifests_are_exactly_nested_and_deterministic():
    samples = records(180)
    first = MODULE.nested_train_manifests(samples, [50, 100, 180], 42, "book")
    second = MODULE.nested_train_manifests(samples, [50, 100, 180], 42, "book")

    assert first == second
    assert set(first["50"]["ids"]) < set(first["100"]["ids"])
    assert set(first["100"]["ids"]) < set(first["180"]["ids"])
    assert len(first["50"]["ids"]) == 50
    assert len(first["100"]["ids"]) == 100
    assert len(first["180"]["ids"]) == 180
    assert first["full"]["count"] == 180
    assert set(first["180"]["ids"]) == set(first["full"]["ids"])


def test_nested_train_manifests_allow_larger_full_pool():
    samples = records(250)
    manifests = MODULE.nested_train_manifests(
        samples, [50, 100, 200], 42, "book"
    )

    assert set(manifests["50"]["ids"]) < set(manifests["100"]["ids"])
    assert set(manifests["100"]["ids"]) < set(manifests["200"]["ids"])
    assert set(manifests["200"]["ids"]) < set(manifests["full"]["ids"])
    assert manifests["full"]["count"] == 250


def test_fixed_candidate_selection_is_deterministic():
    candidates = records(250)
    first = MODULE.select_fixed_candidates(candidates, 60, 42, "book", "test")
    second = MODULE.select_fixed_candidates(candidates, 60, 42, "book", "test")

    assert first == second
    assert len(first) == 60
    assert len({row["id"] for row in first}) == 60
