"""Synthetic regression tests; no dataset files or labels are opened."""
import json
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from converter_core import (compute_metrics, fit_converter, predict_converter,
                            full_ridge_path, low_rank_path, ridge_path, speaker_fallback_mask, stateless_transform)


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1937)

    def test_affine_permuted_columns_and_constant(self):
        X = self.rng.normal(size=(70, 9))
        Y = np.column_stack([3 * X[:, 7] + 4, -2 * X[:, 2] + 1,
                             X[:, 7] * .2, np.full(70, 8)])
        model = fit_converter(X, Y, "best_column_affine")
        assert_array_equal(model["selected_indices"], [7, 2, 7, -1])
        self.assertEqual(model["duplicate_selected_columns"], 1)
        assert_allclose(predict_converter(model, X), Y, atol=1e-12)
        test = self.rng.normal(size=(5, 9))
        expected = np.column_stack([3 * test[:, 7] + 4, -2 * test[:, 2] + 1,
                                    test[:, 7] * .2, np.full(5, 8)])
        assert_allclose(predict_converter(model, test), expected, atol=1e-12)

    def test_exact_column_reorder(self):
        X = self.rng.normal(size=(40, 12))
        Y = X[:, [10, 3, 6]]
        model = fit_converter(X, Y, "best_column_exact")
        assert_array_equal(predict_converter(model, X), Y)
        self.assertEqual(model["exact_train_column_count"], 3)

    def test_exact_hash_precedes_correlation_with_signed_zero_canonicalization(self):
        base = np.arange(20, dtype=np.float64)
        X = np.column_stack([2 * base + 10, base, np.zeros(20)])
        Y = np.column_stack([base, np.full(20, -0.0), 3 * base + 20])
        model = fit_converter(X, Y, "best_column_exact")
        assert_array_equal(model["selected_indices"][:2], [1, 2])
        assert_array_equal(model["exact_train_column_mask"], [True, True, False])
        self.assertEqual(model["exact_train_column_count"], 2)
        assert_array_equal(predict_converter(model, X)[:, :2], Y[:, :2])
        self.assertFalse(np.array_equal(predict_converter(model, X)[:, 2], Y[:, 2]))
        self.assertIn("uncorrected candidate", model["matching_rule"])

    def test_diagonal_affine_scalers_train_only(self):
        X = self.rng.normal(size=(40, 4))
        Y = X * np.array([2, 3, -1, .5]) + np.array([1, 2, 7, 3])
        holdout = self.rng.normal(size=(7, 4)) + 1000
        model = fit_converter(X, Y, "diagonal_affine", groups=np.repeat(range(4), 10))
        before = {key: model[key].copy() for key in ("x_mean", "x_std", "y_mean", "y_std")}
        predict_converter(model, holdout)
        for key in before:
            assert_array_equal(model[key], before[key])
        assert_allclose(model["x_mean"], X.mean(axis=0))
        self.assertEqual(model["n_fit_rows"], 40)
        self.assertEqual(model["n_fit_groups"], 4)
        assert_allclose(predict_converter(model, X), Y, atol=1e-12)

    def test_unseen_speaker_fallback_is_explicit(self):
        X = self.rng.normal(size=(40, 3))
        speakers = np.repeat(["speaker_a", "speaker_b"], 20)
        Y = X * np.where((speakers == "speaker_a")[:, None], 2, 4) + 1
        model = fit_converter(X, Y, "speaker_diagonal_affine", speakers=speakers)
        test = self.rng.normal(size=(3, 3))
        output = predict_converter(model, test, speakers=["speaker_a", "new", "speaker_b"])
        assert_allclose(output[0], 2 * test[0] + 1, atol=1e-12)
        assert_allclose(output[1], test[1] * model["slope"] + model["intercept"])
        assert_allclose(output[2], 4 * test[2] + 1, atol=1e-12)
        assert_array_equal(speaker_fallback_mask(model, ["speaker_a", "new", "speaker_b"]), [False, True, False])

    def test_full_rank_ridge_matches_sklearn(self):
        from sklearn.linear_model import Ridge
        X = self.rng.normal(size=(100, 7))
        Y = self.rng.normal(size=(100, 3))
        models = ridge_path(X, Y, [0, .5, 10], rank=7)
        for model in models:
            xz = (X - X.mean(axis=0)) / X.std(axis=0)
            yz = (Y - Y.mean(axis=0)) / Y.std(axis=0)
            fitted = Ridge(alpha=model["alpha"], fit_intercept=False, solver="svd").fit(xz, yz)
            expected = fitted.predict(xz) * Y.std(axis=0) + Y.mean(axis=0)
            assert_allclose(predict_converter(model, X), expected, atol=3e-6, rtol=3e-6)
        self.assertIs(models[0]["components"], models[1]["components"])

    def test_truncated_pcr_is_ols_in_retained_train_subspace(self):
        X = self.rng.normal(size=(60, 9))
        Y = self.rng.normal(size=(60, 3))
        model = fit_converter(X, Y, "pcr", rank=4)
        latent = ((X - model["x_mean"]) / model["x_scale"]) @ model["components"].T
        yz = (Y - model["y_mean"]) / model["y_scale"]
        coef = np.linalg.lstsq(latent, yz, rcond=None)[0]
        expected = (latent @ coef) * model["y_scale"] + model["y_mean"]
        assert_allclose(predict_converter(model, X), expected, atol=3e-6, rtol=3e-6)

    def test_full_ridge_matches_sklearn_with_constants_and_collinearity(self):
        from sklearn.linear_model import Ridge
        X = self.rng.normal(size=(60, 12))
        X[:, 10], X[:, 11] = X[:, 0] * 2, 17.0
        Y = self.rng.normal(size=(60, 4))
        Y[:, 3] = 9
        test = self.rng.normal(size=(7, 12))
        test[:, 10], test[:, 11] = test[:, 0] * 2, 17.0
        models = full_ridge_path(X, Y, alphas=[.01, 1., 100.], groups=np.repeat(range(3), 20))
        for model in models:
            xz = (X - model["x_mean"]) / model["x_scale"]
            yz = (Y - model["y_mean"]) / model["y_scale"]
            fitted = Ridge(alpha=model["alpha"], fit_intercept=False, solver="svd").fit(xz, yz)
            test_z = (test - model["x_mean"]) / model["x_scale"]
            expected = fitted.predict(test_z) * model["y_scale"] + model["y_mean"]
            assert_allclose(predict_converter(model, test), expected, atol=1e-11, rtol=1e-11)
            assert_array_equal(predict_converter(model, test)[:, 3], np.full(7, 9.))
            self.assertEqual(model["coef"].shape, (12, 4))
            self.assertEqual(model["n_fit_groups"], 3)
        self.assertIs(models[0]["x_mean"], models[1]["x_mean"])

    def test_full_ridge_scalers_train_only_and_no_prediction_mutation(self):
        X = self.rng.normal(size=(30, 8))
        Y = self.rng.normal(size=(30, 3))
        model = fit_converter(X, Y, "full_ridge", alpha=1.)
        before = {key: model[key].copy() for key in ("x_mean", "x_scale", "y_mean", "y_scale")}
        predict_converter(model, np.full((5, 8), 10000.))
        assert_allclose(model["x_mean"], X.mean(axis=0), atol=1e-15)
        assert_allclose(model["y_mean"], Y.mean(axis=0), atol=1e-15)
        for key in before:
            assert_array_equal(model[key], before[key])

    def test_rank_paths_share_one_train_basis(self):
        X = self.rng.normal(size=(60, 9))
        Y = self.rng.normal(size=(60, 3))
        models = low_rank_path(X, Y, ranks=[2, 4, 8], alphas=[0, 1])
        self.assertEqual(len(models), 6)
        assert_array_equal(models[0]["components"], models[-1]["components"][:2])
        for model in models:
            self.assertEqual(model["svd_basis_rank"], 8)
            self.assertEqual(model["random_state"], 424242)
            self.assertEqual(predict_converter(model, X).shape, Y.shape)

    def test_global_and_dialogue_equal_nrmse_definitions(self):
        Y = np.zeros((3, 2))
        P = np.array([[0, 0], [0, 4], [2, 2]])
        summary, arrays = compute_metrics(Y, P, target_scale=np.array([1, 2]),
                                          groups=[0, 0, 1], return_per_dimension=True)
        self.assertAlmostEqual(summary["global_train_scale_nrmse"], np.sqrt(1.5))
        assert_allclose(arrays["dialogue_nrmse"], [1, np.sqrt(2.5)])
        self.assertAlmostEqual(summary["dialogue_equal_nrmse"], (1 + np.sqrt(2.5)) / 2)
        self.assertNotAlmostEqual(summary["global_train_scale_nrmse"], summary["train_scale_nrmse_mean"])

    def test_nonfinite_and_zero_metrics(self):
        Y = np.array([[0, 0], [1, 2], [np.nan, 3], [1, 2]])
        P = np.array([[0, 0], [1, 2], [4, 4], [2, 4]])
        summary, arrays = compute_metrics(Y, P, target_scale=np.array([1., 0.]),
                                          groups=[0, 0, 1, 1], return_per_dimension=True,
                                          wasserstein=True)
        self.assertEqual(summary["nonfinite_target_elements"], 1)
        self.assertEqual(summary["nonfinite_rows"], 1)
        self.assertEqual(summary["cosine_undefined_finite_rows"], 1)
        self.assertEqual(summary["cosine_valid_rows"], 2)
        self.assertEqual(summary["max_absolute_error"], 2)
        self.assertAlmostEqual(summary["mean_absolute_error"], 4 / 7)
        self.assertTrue(np.isnan(arrays["column_train_scale_nrmse"][1]))
        json.dumps(summary, allow_nan=False)
        all_nan = compute_metrics(np.array([[np.nan]]), np.array([[np.nan]]))
        self.assertIsNone(all_nan["mean_absolute_error"])
        json.dumps(all_nan, allow_nan=False)

    def test_row_transforms_and_reject_training_nan(self):
        X = np.array([[1, 2, 3], [4, 4, 4.]])
        assert_allclose(stateless_transform(X, "rowminmax"), [[0, .5, 1], [0, 0, 0]])
        assert_allclose(stateless_transform(X, "row_minmax"), stateless_transform(X, "rowminmax"))
        assert_allclose(stateless_transform(X, "row_zscore"), stateless_transform(X, "rowzscore"))
        assert_allclose(np.linalg.norm(stateless_transform(X, "row_l2"), axis=1), [1, 1])
        assert_array_equal(stateless_transform(np.zeros((1, 3)), "row_l2"), np.zeros((1, 3)))
        assert_allclose(stateless_transform(X, "rowzscore").mean(axis=1), 0, atol=1e-15)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            fit_converter(np.array([[np.nan]]), np.array([[1]]), "identity")


if __name__ == "__main__":
    unittest.main(verbosity=2)
