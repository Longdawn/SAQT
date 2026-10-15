import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "prepare_curated_historical_columns.py"
SPEC = importlib.util.spec_from_file_location("prepare_curated_historical_columns", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def character(text, x1, y1, x2, y2):
    return {
        "text": text,
        "box": (x1, y1, x2, y2),
        "cx": (x1 + x2) / 2,
        "cy": (y1 + y2) / 2,
        "width": x2 - x1,
        "height": y2 - y1,
    }


def test_group_vertical_columns_uses_right_to_left_and_top_to_bottom_order():
    characters = [
        character("下", 100, 40, 120, 60),
        character("右", 100, 10, 120, 30),
        character("列", 40, 40, 60, 60),
        character("左", 40, 10, 60, 30),
    ]

    columns = MODULE.group_vertical_columns(characters, center_gap_factor=0.6)

    assert [column["text"] for column in columns] == ["右下", "左列"]
    assert [column["center_x"] for column in columns] == [110, 50]


def test_split_pages_is_deterministic_and_disjoint():
    pages = [f"page_{index:03d}" for index in range(20)]
    ratios = {"train": 0.7, "valid": 0.1, "test": 0.2}

    first = MODULE.split_pages(pages, ratios, 42, "source_a")
    second = MODULE.split_pages(pages, ratios, 42, "source_a")

    assert first == second
    assert {split: len(values) for split, values in first.items()} == {
        "train": 14,
        "valid": 2,
        "test": 4,
    }
    assert set(first["train"]).isdisjoint(first["valid"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["valid"]).isdisjoint(first["test"])
    assert set().union(*map(set, first.values())) == set(pages)


def test_allocate_quota_preserves_total():
    quotas = MODULE.allocate_quota(
        350, {"train": 0.7, "valid": 0.1, "test": 0.2}
    )

    assert quotas == {"train": 245, "valid": 35, "test": 70}
    assert sum(quotas.values()) == 350


def test_weighted_page_split_is_deterministic_and_page_disjoint():
    weights = {f"page_{index:02d}": 10 + index for index in range(12)}
    ratios = {"train": 0.7, "valid": 0.1, "test": 0.2}

    first = MODULE.split_weighted_pages(weights, ratios, 42, "source_a")
    second = MODULE.split_weighted_pages(weights, ratios, 42, "source_a")

    assert first == second
    assert all(first[split] for split in MODULE.SPLITS)
    assert set().union(*map(set, first.values())) == set(weights)
    for split in MODULE.SPLITS:
        other_pages = set().union(
            *(set(first[other]) for other in MODULE.SPLITS if other != split)
        )
        assert set(first[split]).isdisjoint(other_pages)


def test_column_eligibility_rejects_unknown_labels():
    catalog = {"min_column_chars": 2, "disallowed_labels": ["■", "#"]}

    assert MODULE.column_is_eligible({"text": "古籍"}, catalog)
    assert not MODULE.column_is_eligible({"text": "古■"}, catalog)
    assert not MODULE.column_is_eligible({"text": "古"}, catalog)


def test_drop_isolated_leading_char_removes_only_large_first_gap():
    column = MODULE.make_column(
        [
            character("首", 10, 0, 30, 20),
            character("正", 10, 90, 30, 110),
            character("文", 10, 120, 30, 140),
            character("列", 10, 150, 30, 170),
        ]
    )

    trimmed, dropped = MODULE.drop_isolated_leading_char(column, 2.5)

    assert dropped
    assert trimmed["text"] == "正文列"
    assert trimmed["box"] == (10, 90, 30, 170)


def test_drop_isolated_leading_char_preserves_regular_column():
    column = MODULE.make_column(
        [
            character("古", 10, 0, 30, 20),
            character("籍", 10, 30, 30, 50),
            character("文", 10, 60, 30, 80),
        ]
    )

    preserved, dropped = MODULE.drop_isolated_leading_char(column, 2.5)

    assert not dropped
    assert preserved["text"] == "古籍文"
