import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
MODULE = ROOT / 'experiments/xp_caldist_behavioral_calibration/formal_fivefold_fiveseed/20261109T174024Z/scripts/calibration_only.py'

def load():
    spec = importlib.util.spec_from_file_location('formal_calibration_target', MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_global_fit_records_complete_objective_trace():
    mod = load()
    logits = np.array([[2., 0., -1.], [0., 2., -1.], [1., .5, 0.], [.2, .3, .4]])
    labels = np.array([0, 1, 2, 2])
    fit = mod.fit_global(logits, labels, maxiter=50)
    assert fit['fit_curve']
    assert all(set(row) == {'evaluation', 'theta', 'temperature', 'objective'} for row in fit['fit_curve'])
    assert all(np.isfinite(list(row.values())).all() for row in fit['fit_curve'])

def test_classwise_fit_records_objective_gradient_and_temperatures():
    mod = load()
    logits = np.array([[2., 0., -1.], [0., 2., -1.], [1., .5, 0.], [.2, .3, .4]])
    labels = np.array([0, 1, 2, 2])
    selector = logits.argmax(1)
    fit = mod.fit_predicted_class(logits, labels, selector, 0.0, 0.1, maxiter=50)
    assert fit['fit_curve']
    for row in fit['fit_curve']:
        assert set(row) == {'evaluation', 'objective', 'gradient_norm', 'log_temperatures', 'temperatures'}
        assert np.isfinite(row['objective'])
        assert np.isfinite(row['gradient_norm'])
        assert len(row['temperatures']) == len(np.unique(selector))
