"""Scientific correctness checks for offset fitting and nested data isolation."""

import unittest
import numpy as np
from scipy.optimize import check_grad
from scipy.special import expit

from calibration_core import (base_logits, correction_logits, fit_base, fit_correction,
                              fit_nested_fold, offset_objective, predict)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1475)
        self.c = rng.uniform(.2, .95, 96)
        self.b = rng.normal(size=(96, 4))
        self.y = rng.binomial(1, self.c)
        self.groups = np.repeat(np.arange(12).astype(str), 8)

    def test_analytic_gradient(self):
        w = np.array([.3, -.1, .5, -.8])
        offset = self.c * 2 - 1
        error = check_grad(lambda z: offset_objective(z, self.b, offset, self.y, .1)[0],
                           lambda z: offset_objective(z, self.b, offset, self.y, .1)[1], w)
        self.assertLess(error, 1e-6)

    def test_off_exactly_recovers_base(self):
        for name in ["E1", "L0"]:
            base = fit_base(self.c, self.y, name)
            offset = base_logits(self.c, base)
            correction = fit_correction(self.b, offset, self.y, "off")
            np.testing.assert_array_equal(expit(correction_logits(self.b, offset, correction)), expit(offset))

    def test_parameters_invariant_to_outer_held_data(self):
        train = np.arange(80)
        held = np.arange(80, 96)
        fit = fit_nested_fold(self.c[train], self.b[train], self.y[train], self.groups[train], "E1")
        c2, b2, y2 = self.c.copy(), self.b.copy(), self.y.copy()
        c2[held], b2[held], y2[held] = .999, 1e6, 1 - y2[held]
        changed = fit_nested_fold(c2[train], b2[train], y2[train], self.groups[train], "E1")
        self.assertEqual(fit, changed)
        self.assertEqual(set(fit["inner_folds"][0]["train_dialogues"]) &
                         set(fit["inner_folds"][0]["held_dialogues"]), set())
        self.assertTrue(np.isfinite(predict(self.c[held], self.b[held], fit)).all())

    def test_offset_optimizer_stationarity(self):
        fitted = fit_correction(self.b, np.zeros(len(self.y)), self.y, .1)
        self.assertLess(fitted["gradient_max_abs"], 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
