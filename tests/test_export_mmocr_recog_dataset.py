import argparse

from tools.export_mmocr_recog_dataset import selected_exports


def args(name=None, roots=None, preset="all"):
    return argparse.Namespace(name=name, roots=roots, preset=preset)


def test_custom_export_takes_precedence_over_preset():
    assert selected_exports(args("pilot", ["book_a"])) == [
        ("pilot", ["book_a"])
    ]


def test_custom_export_requires_name_and_roots_together():
    for value in (args(name="pilot"), args(roots=["book_a"])):
        try:
            selected_exports(value)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected incomplete custom export to fail")
