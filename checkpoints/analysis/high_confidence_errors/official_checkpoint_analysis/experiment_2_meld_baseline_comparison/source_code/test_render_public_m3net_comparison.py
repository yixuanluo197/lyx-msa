"""Protect sparse-bin plotting and CSV-to-Word verification for the new report."""
from pathlib import Path
import tempfile
import unittest

from docx import Document

from scripts.render_public_m3net_comparison import (
    display, make_figure, make_word, verify_word, EMOTIONS)


def fixture_sources():
    def source(rows):
        return list(rows[0]), rows
    paired = dict(dataset='IEMOCAP', dialogue_id='dialogue', utterance_index='0')
    hc = dict(threshold='0.9', baseline_high_confidence_predictions='3', baseline_high_confidence_errors='2',
        baseline_high_confidence_error_rate=str(2/3), cmerc_high_confidence_predictions='2',
        cmerc_high_confidence_errors='1', cmerc_high_confidence_error_rate='.5', corrected='1',
        still_wrong_confidence_lower='1', still_wrong_confidence_not_lower='0',
        still_wrong_above_threshold='1', new_cmerc_hc_errors='0', corrected_ratio='.5',
        still_wrong_confidence_lower_ratio='.5', still_wrong_confidence_not_lower_ratio='0',
        still_wrong_above_threshold_ratio='.5', eligible_non_baseline_hc_errors='1621',
        new_cmerc_hc_error_ratio_of_eligible='0', new_cmerc_hc_errors_from_baseline_correct='0',
        new_cmerc_hc_errors_from_baseline_wrong_below_threshold='0')
    classwise = dict(true_emotion='sad', support='3', baseline_errors='2', cmerc_errors='1',
        corrected='1', regressed='0', both_wrong='1',
        net_error_change='-1')
    metrics = [dict(model=model, valid_utterances='1623', total_errors='456', accuracy='.7',
        weighted_f1='.7', mean_confidence='.8', mean_error_confidence='.75',
        ece='.1', mce='.2', brier_score='.4', nll='.9') for model in ('Official M³Net', 'Official CMERC')]
    bins = []
    for model in ('Official M³Net', 'Official CMERC'):
        for i in range(10):
            bins.append(dict(model=model, bin_index=str(i), lower_bound=str(i/10),
                upper_bound=str((i+1)/10), count='1623' if i == 9 else '0',
                mean_confidence='.95' if i == 9 else '', accuracy='.7' if i == 9 else '',
                signed_gap='.25' if i == 9 else ''))
    return {'paired_predictions.csv': source([paired]),
        'classification_transitions.csv': source([dict(transition='both_correct', count='3', ratio='1')]),
        'high_confidence_transitions.csv': source([hc]),
        'classwise_transitions.csv': source([classwise]),
        'calibration_comparison.csv': source(metrics),
        'calibration_bins.csv': source(bins), 'typical_cases.csv': (['case_type', 'threshold', 'selection_score'], [])}


class PublicReportTest(unittest.TestCase):
    def test_empty_bin_is_na_not_a_false_zero(self):
        self.assertEqual(display('', percent=True), 'N/A')
        self.assertEqual(display('-.01', percent=True, signed=True), '-1.0000')
        with self.assertRaises(ValueError):
            display('nan')
        with self.assertRaises(ValueError):
            display('1.5', integer=True)

    def test_empty_bins_never_generate_reliability_points(self):
        sources = fixture_sources()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'figures').mkdir()
            figure = make_figure(path, sources['calibration_bins.csv'][1])
            for series in figure['model_series']:
                self.assertEqual(series['nonempty_bin_indices'], [9])
                self.assertEqual(series['points'], [[.95, .7]])
            self.assertTrue((path / 'figures/reliability_IEMOCAP_public_comparison.svg').is_file())

    def test_word_verifier_rejects_corrupted_numeric_cell(self):
        sources = fixture_sources()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'figures').mkdir()
            make_figure(path, sources['calibration_bins.csv'][1])
            records, headings = make_word(path, sources)
            from scripts.render_public_m3net_comparison import DOCX_NAME, FIGURE_STEM
            docx = path / DOCX_NAME
            result = verify_word(docx, sources, records, headings, path / f'{FIGURE_STEM}.png')
            self.assertTrue(result['passed'])
            document = Document(docx)
            document.tables[0].rows[1].cells[1].text = '999'
            document.save(docx)
            with self.assertRaisesRegex(AssertionError, 'differs from source CSV'):
                verify_word(docx, sources, records, headings, path / f'{FIGURE_STEM}.png')


if __name__ == '__main__':
    unittest.main()
