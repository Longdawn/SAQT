import json
import math
import tempfile
import unittest
from pathlib import Path

from finetuning import _best_cer_from_log, _resolve_resume_progress


class FinetuningResumeTest(unittest.TestCase):
    def test_exact_output_resume_preserves_progress_and_logged_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "log.txt"
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps({"test_cer_oracle_direction": 0.12}),
                        "not-json",
                        json.dumps({"test_cer_oracle_direction": 0.096}),
                        json.dumps({"test_cer_oracle_direction": 0.103}),
                    ]
                ),
                encoding="utf-8",
            )

            start_epoch, global_step, best_cer = _resolve_resume_progress(
                {"epoch": 9, "global_step": 65000},
                resume_from_output_checkpoint=True,
                new_class_embedding=True,
                log_path=log_path,
            )

        self.assertEqual(start_epoch, 10)
        self.assertEqual(global_step, 65000)
        self.assertAlmostEqual(best_cer, 0.096)

    def test_checkpoint_best_takes_precedence_over_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "log.txt"
            log_path.write_text(
                json.dumps({"test_cer_oracle_direction": 0.08}),
                encoding="utf-8",
            )
            _, _, best_cer = _resolve_resume_progress(
                {"epoch": 3, "global_step": 20, "best_cer": 0.09},
                resume_from_output_checkpoint=True,
                new_class_embedding=True,
                log_path=log_path,
            )

        self.assertAlmostEqual(best_cer, 0.09)

    def test_external_head_checkpoint_starts_new_embedding_run_at_zero(self):
        start_epoch, global_step, best_cer = _resolve_resume_progress(
            {"epoch": 2, "global_step": 15000},
            resume_from_output_checkpoint=False,
            new_class_embedding=True,
            log_path="missing.log",
        )

        self.assertEqual(start_epoch, 0)
        self.assertEqual(global_step, 15000)
        self.assertTrue(math.isinf(best_cer))

    def test_best_cer_reader_ignores_invalid_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "log.txt"
            log_path.write_text(
                '{"test_cer": 0.2}\n{}\ninvalid\n{"test_cer": 0.1}\n',
                encoding="utf-8",
            )
            self.assertAlmostEqual(_best_cer_from_log(log_path), 0.1)


if __name__ == "__main__":
    unittest.main()
