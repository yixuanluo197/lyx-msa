"""Check report-specific preservation of numeric, text and empty CSV inputs."""
import math
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.render_official_reliability import (
    excel_value, make_excel, verify_excel, formatted,
)


class OfficialReportTest(unittest.TestCase):
    def test_workbook_preserves_probabilities_ids_and_empty_error_schema(self):
        sources = {
            'official_cmerc_summary.csv': (
                ['dataset', 'accuracy', 'ece'],
                [{'dataset': 'MELD', 'accuracy': '0.6777777777777778',
                  'ece': '0.2281759267790986'}]),
            'MELD/all_predictions.csv': (
                ['dialogue_id', 'utterance_index', 'utterance_text', 'probability_neutral'],
                [{'dialogue_id': '00012', 'utterance_index': '3',
                  'utterance_text': '=1+1', 'probability_neutral': '0.00000000000000012345'}]),
            'MELD/high_confidence_errors.csv': (['dataset', 'threshold'], []),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.xlsx'
            sheets = make_excel(path, sources)
            result = verify_excel(path, sources, sheets)
            self.assertTrue(result['passed'])
            self.assertEqual(result['worksheets'][-1]['row_count'], 0)
            self.assertEqual(len(sheets), 3)
            with zipfile.ZipFile(path) as archive:
                self.assertNotIn(b'<f>', archive.read('xl/worksheets/sheet3.xml'))

    def test_missing_rates_are_not_zero_and_text_ids_are_not_numbers(self):
        self.assertEqual(formatted('', percent=True), 'N/A')
        self.assertEqual(excel_value('dialogue_id', '0012'), '0012')
        self.assertEqual(excel_value('high_confidence_error_rate_ge_0p9', ''), '')
        self.assertTrue(math.isclose(excel_value('probability_neutral', '0.25'), .25))
        self.assertEqual(excel_value('full_correct', 'False'), 0)


if __name__ == '__main__':
    unittest.main()
