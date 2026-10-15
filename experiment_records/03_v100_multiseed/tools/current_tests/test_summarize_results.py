import math
import unittest

from scripts.summarize_results import best_training_record, paper_reference


class TrainingSummaryTests(unittest.TestCase):
    def test_official_protocol_uses_highest_test_selected_score(self):
        records = [
            {"epoch": 1, "selection_weighted_f1": 60.0, "test_weighted_f1": 60.0},
            {"epoch": 2, "selection_weighted_f1": 59.0, "test_weighted_f1": 59.0},
        ]
        self.assertEqual(best_training_record(records, "official_test")["epoch"], 1)

    def test_validation_protocol_ignores_nan_test_fields(self):
        records = [
            {"epoch": 1, "selection_weighted_f1": 50.0, "test_weighted_f1": None},
            {"epoch": 2, "selection_weighted_f1": 55.0, "test_weighted_f1": None},
        ]
        result = best_training_record(records, "validation")
        self.assertEqual(result["epoch"], 2)
        self.assertTrue(math.isclose(result["selection_weighted_f1"], 55.0))

    def test_paper_reference_distinguishes_baseline_and_cmerc(self):
        self.assertEqual(paper_reference("IEMOCAP", "baseline"), {"weighted_f1": 69.61, "cel": 1772.54})
        self.assertEqual(paper_reference("MELD", "cmerc"), {"weighted_f1": 66.85, "cel": 605.05})


if __name__ == "__main__":
    unittest.main()
