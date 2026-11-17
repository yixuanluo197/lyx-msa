"""Train-only, explicitly reconstructed acoustic feature converters.

This module never reads files, labels, or splits. Callers must supply only the
effective training fold to ``fit_converter``. All stored statistics come from
that input. No method is represented as an author's original transformation.
"""
from __future__ import annotations

import numpy as np
import hashlib


def _matrix(value, name):
    value = np.asarray(value)
    if value.ndim != 2 or not value.shape[0] or not value.shape[1]:
        raise ValueError(f"{name} must be a nonempty 2D matrix")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite entries; no rows were dropped")
    return value


def _moments(value, chunk=256):
    mean = np.empty(value.shape[1], dtype=np.float64)
    std = np.empty_like(mean)
    for start in range(0, value.shape[1], chunk):
        block = np.asarray(value[:, start:start + chunk], dtype=np.float64)
        mean[start:start + chunk] = block.mean(axis=0)
        std[start:start + chunk] = block.std(axis=0)
    return mean, std


def stateless_transform(X, kind="none"):
    """Per-row transforms, with constant rows mapping to zero (no fitted state)."""
    X = _matrix(X, "X")
    kind = {"row_minmax": "rowminmax", "row_zscore": "rowzscore"}.get(kind, kind)
    if kind == "none":
        return X
    X = np.asarray(X, dtype=np.float64)
    if kind == "rowminmax":
        center = X.min(axis=1, keepdims=True)
        scale = X.max(axis=1, keepdims=True) - center
    elif kind == "rowzscore":
        center = X.mean(axis=1, keepdims=True)
        scale = X.std(axis=1, keepdims=True)
    elif kind == "row_l2":
        center = 0.0
        scale = np.linalg.norm(X, axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown stateless transform: {kind}")
    return (X - center) / np.where(scale > 0, scale, 1.0)


def _affine(X, Y):
    xm, xs = _moments(X)
    ym, ys = _moments(Y)
    covariance = np.empty(X.shape[1], dtype=np.float64)
    for start in range(0, X.shape[1], 256):
        xb = np.asarray(X[:, start:start + 256], dtype=np.float64) - xm[start:start + 256]
        yb = np.asarray(Y[:, start:start + 256], dtype=np.float64) - ym[start:start + 256]
        covariance[start:start + 256] = np.mean(xb * yb, axis=0)
    slope = np.divide(covariance, xs ** 2, out=np.zeros_like(xs), where=xs > 0)
    return {"slope": slope, "intercept": ym - slope * xm,
            "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys}


def _best_columns(X, Y, affine):
    xm, xs = _moments(X)
    ym, ys = _moments(Y)
    yc = np.asarray(Y, dtype=np.float64) - ym
    score = np.full(Y.shape[1], -np.inf)
    index = np.full(Y.shape[1], -1, dtype=np.int64)
    signed_correlation = np.zeros(Y.shape[1], dtype=np.float64)
    # Never materialize the 6373 x 1611 full correlation matrix. Workspace is
    # centered Y plus a 128-column X block and a 128 x output-width product.
    for start in range(0, X.shape[1], 128):
        stop = min(start + 128, X.shape[1])
        xb = np.asarray(X[:, start:stop], dtype=np.float64) - xm[start:stop]
        corr = (xb.T @ yc) / X.shape[0]
        denominator = xs[start:stop, None] * ys[None, :]
        np.divide(corr, denominator, out=corr, where=denominator > 0)
        corr[denominator == 0] = 0
        local = np.abs(corr).argmax(axis=0)
        columns = np.arange(Y.shape[1])
        local_score = np.abs(corr[local, columns])
        better = local_score > score  # deterministic lower-column tie break
        index[better] = start + local[better]
        signed_correlation[better] = corr[local[better], columns[better]]
        score[better] = local_score[better]
    exact_matches = np.zeros(Y.shape[1], dtype=bool)
    if not affine:
        # Canonical float64 values, normalized signed zero, and explicit value
        # verification after hashing. Hash identity is a search acceleration,
        # never a substitute for checking whole columns.
        def canonical_column(matrix, j):
            column = np.array(matrix[:, j], dtype="<f8", order="C", copy=True)
            column[column == 0] = 0.0
            return column

        lookup = {}
        for j in range(X.shape[1]):
            digest = hashlib.sha256(canonical_column(X, j).tobytes()).digest()
            lookup.setdefault(digest, []).append(j)
        for j in range(Y.shape[1]):
            column = canonical_column(Y, j)
            digest = hashlib.sha256(column.tobytes()).digest()
            for candidate in lookup.get(digest, []):
                if np.array_equal(canonical_column(X, candidate), column):
                    index[j] = candidate
                    exact_matches[j] = True
                    signed_correlation[j] = 1.0 if ys[j] > 0 else 0.0
                    break
    constant = (ys == 0) & ~exact_matches
    index[constant] = -1
    slope = np.ones(Y.shape[1], dtype=np.float64)
    intercept = np.zeros(Y.shape[1], dtype=np.float64)
    if affine:
        valid = ~constant
        slope[valid] = np.divide(signed_correlation[valid] * ys[valid], xs[index[valid]],
                                 out=np.zeros(valid.sum()), where=xs[index[valid]] > 0)
        intercept[valid] = ym[valid] - slope[valid] * xm[index[valid]]
    slope[constant] = 0
    intercept[constant] = ym[constant]
    chosen = index[index >= 0]
    return {"selected_indices": index, "slope": slope, "intercept": intercept,
            "signed_training_correlations": signed_correlation,
            "duplicate_selected_columns": int(len(chosen) - len(np.unique(chosen))),
            "constant_target_columns": int((ys == 0).sum()),
            "exact_train_column_count": int(exact_matches.sum()) if not affine else None,
            "exact_train_column_mask": exact_matches if not affine else None,
            "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys,
            "matching_rule": ("maximum absolute train Pearson then OLS rescaling/offset; lower index breaks ties"
                              if affine else "canonical float64 whole-column SHA256 then exact numeric verification first; "
                              "fallback maximum absolute train Pearson without correcting sign, scale, or offset; "
                              "fallback is an uncorrected candidate, not an exact recovered column; lower index breaks ties")}


def _low_rank_basis(X, Y, rank, random_state=424242):
    from sklearn.utils.extmath import randomized_svd
    xm, xs = _moments(X)
    ym, ys = _moments(Y)
    xscale = np.where(xs > 0, xs, 1.0)
    yscale = np.where(ys > 0, ys, 1.0)
    # Float32 working arrays limit memory for 6373-column inputs. Fitted means,
    # scales and final metrics remain float64. Predictions use stored factors.
    xz = np.asarray(X, dtype=np.float32).copy()
    xz -= xm.astype(np.float32)
    xz /= xscale.astype(np.float32)
    rank = min(int(rank), X.shape[0] - 1, X.shape[1])
    if rank < 1:
        raise ValueError("low-rank converters need at least two training rows")
    u, s, vt = randomized_svd(xz, n_components=rank, n_iter=4,
                              random_state=random_state, flip_sign=True)
    del xz
    yz = (np.asarray(Y, dtype=np.float64) - ym) / yscale
    projected_y = u.astype(np.float64).T @ yz
    return {"x_mean": xm, "x_std": xs, "x_scale": xscale,
            "y_mean": ym, "y_std": ys, "y_scale": yscale,
            "components": vt, "singular_values": s.astype(np.float64),
            "projected_y": projected_y, "rank": rank,
            "singular_cutoff": float(np.finfo(np.float32).eps * max(X.shape) * s[0]),
            "random_state": random_state, "svd_working_dtype": "float32"}


def ridge_path(X, Y, alphas, rank=128, groups=None):
    """Fit train-standardized truncated-SVD ridge models sharing one SVD.

    Objective: ||Zx B - Zy||_F^2 + alpha ||B||_F^2 in the retained subspace.
    alpha is not divided by the row count. ``groups`` is validation metadata:
    no group split or hyperparameter selection occurs inside this function.
    """
    X, Y = _matrix(X, "X"), _matrix(Y, "Y")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y row counts differ")
    basis = _low_rank_basis(X, Y, rank)
    projected_y = basis.pop("projected_y")
    s = basis["singular_values"]
    models = []
    for alpha in alphas:
        alpha = float(alpha)
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        usable = s > basis["singular_cutoff"] if alpha == 0 else (s * s + alpha) > 0
        multiplier = np.divide(s, s * s + alpha, out=np.zeros_like(s), where=usable)
        model = {**basis, "kind": "ridge" if alpha else "pcr",
                 "reconstructed_method": True, "alpha": alpha,
                 "latent_coef": multiplier[:, None] * projected_y,
                 "numerical_rank": int((s > basis["singular_cutoff"]).sum()),
                 "n_fit_rows": X.shape[0], "n_input_features": X.shape[1],
                 "n_output_features": Y.shape[1]}
        if groups is not None:
            if len(groups) != X.shape[0]:
                raise ValueError("groups length differs from training rows")
            model["n_fit_groups"] = len(np.unique(groups))
        models.append(model)
    return models


def low_rank_path(X, Y, ranks=(16, 64, 128), alphas=(0.0, 1.0), groups=None):
    """Rank x regularization candidates from ONE train-fold SVD.

    The largest requested rank determines the randomized basis; lower-rank
    candidates reuse its leading components. Each model records that choice.
    """
    ranks = sorted(set(int(rank) for rank in ranks))
    if not ranks or ranks[0] < 1:
        raise ValueError("ranks must contain positive integers")
    full_models = ridge_path(X, Y, alphas, rank=max(ranks), groups=groups)
    models = []
    for requested_rank in ranks:
        for full in full_models:
            rank = min(requested_rank, full["rank"])
            models.append({**full, "rank": rank, "requested_rank": requested_rank,
                           "svd_basis_rank": full["rank"],
                           "components": full["components"][:rank],
                           "singular_values": full["singular_values"][:rank],
                           "latent_coef": full["latent_coef"][:rank],
                           "numerical_rank": int((full["singular_values"][:rank] >
                                                  full["singular_cutoff"]).sum())})
    return models


def full_ridge_path(X, Y, alphas=(0.01, 1.0, 100.0), groups=None):
    """Full-rank multioutput Ridge, with one float64 Gram/cross per fold.

    The strictly positive penalty permits a primal Cholesky solve, including
    constant or collinear input columns. All normalizers use only supplied
    training rows. No PCA/truncation, labels, or held-out rows are accessed.
    The objective is ||Zx B - Zy||_F^2 + alpha ||B||_F^2 (unnormalized sum).
    """
    from scipy.linalg import cho_factor, cho_solve
    X, Y = _matrix(X, "X"), _matrix(Y, "Y")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y row counts differ")
    if groups is not None and len(groups) != X.shape[0]:
        raise ValueError("groups length differs from training rows")
    xm, xs = _moments(X)
    ym, ys = _moments(Y)
    xscale, yscale = np.where(xs > 0, xs, 1.0), np.where(ys > 0, ys, 1.0)
    xz = np.array(X, dtype=np.float64, copy=True)
    xz -= xm
    xz /= xscale
    yz = np.array(Y, dtype=np.float64, copy=True)
    yz -= ym
    yz /= yscale
    gram = xz.T @ xz
    cross = xz.T @ yz
    del xz, yz
    common = {"kind": "full_ridge", "reconstructed_method": True,
              "x_mean": xm, "x_std": xs, "x_scale": xscale,
              "y_mean": ym, "y_std": ys, "y_scale": yscale,
              "n_fit_rows": X.shape[0], "n_input_features": X.shape[1],
              "n_output_features": Y.shape[1], "solver_dtype": "float64",
              "solver": "primal Cholesky; one train-fold Gram/cross reused across alpha grid",
              "objective": "sum((Zx @ coef - Zy)^2) + alpha * sum(coef^2)",
              "projection": "none; full input feature rank retained",
              "shared_gram_shape": list(gram.shape), "shared_cross_shape": list(cross.shape)}
    if groups is not None:
        common["n_fit_groups"] = len(np.unique(groups))
    models = []
    for alpha in alphas:
        alpha = float(alpha)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("full-rank Ridge alpha must be positive and finite")
        lhs = np.array(gram, dtype=np.float64, order="F", copy=True)
        diagonal = np.diag_indices_from(lhs)
        lhs[diagonal] += alpha
        factor = cho_factor(lhs, lower=True, overwrite_a=True, check_finite=False)
        coef = cho_solve(factor, cross, overwrite_b=False, check_finite=False)
        if not np.isfinite(coef).all():
            raise FloatingPointError("nonfinite full-rank Ridge coefficients")
        models.append({**common, "alpha": alpha, "coef": coef})
        del lhs, factor
    return models


def fit_converter(X, Y, kind, groups=None, speakers=None, alpha=1.0, rank=128):
    X, Y = _matrix(X, "X"), _matrix(Y, "Y")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y row counts differ")
    model = {"kind": kind, "reconstructed_method": True,
             "n_fit_rows": X.shape[0], "n_input_features": X.shape[1],
             "n_output_features": Y.shape[1]}
    if groups is not None:
        if len(groups) != X.shape[0]:
            raise ValueError("groups length differs from training rows")
        model["n_fit_groups"] = len(np.unique(groups))
    if kind == "identity":
        if X.shape[1] != Y.shape[1]:
            raise ValueError("identity requires equal dimensions")
    elif kind in ("best_column_exact", "best_column_affine"):
        model.update(_best_columns(X, Y, affine=kind.endswith("affine")))
    elif kind in ("diagonal_affine", "speaker_diagonal_affine"):
        if X.shape[1] != Y.shape[1]:
            raise ValueError("diagonal affine requires equal dimensions")
        model.update(_affine(X, Y))
        if kind == "speaker_diagonal_affine":
            if speakers is None or len(speakers) != X.shape[0]:
                raise ValueError("one speaker key per training row is required")
            speakers = np.asarray(speakers).astype(str)
            model["speaker_parameters"] = {
                key: {**_affine(X[speakers == key], Y[speakers == key]),
                      "n_fit_rows": int((speakers == key).sum())}
                for key in sorted(np.unique(speakers))}
            model["unseen_speaker_policy"] = "global_training_affine_fallback"
    elif kind in ("ridge", "pcr"):
        return ridge_path(X, Y, [alpha if kind == "ridge" else 0.0],
                          rank=rank, groups=groups)[0]
    elif kind == "full_ridge":
        return full_ridge_path(X, Y, [alpha], groups=groups)[0]
    else:
        raise ValueError(f"unknown converter: {kind}")
    return model


def speaker_fallback_mask(model, speakers):
    """Explicitly report rows using fallback; prediction itself returns arrays."""
    if model["kind"] != "speaker_diagonal_affine":
        return np.zeros(len(speakers), dtype=bool)
    return np.asarray([str(key) not in model["speaker_parameters"] for key in speakers])


def predict_converter(model, X, speakers=None):
    X = _matrix(X, "X")
    if X.shape[1] != model["n_input_features"]:
        raise ValueError("input feature width differs from fitted converter")
    kind = model["kind"]
    if kind == "identity":
        return X.copy()
    if kind in ("best_column_exact", "best_column_affine"):
        index = model["selected_indices"]
        return X[:, np.maximum(index, 0)] * model["slope"] + model["intercept"]
    if kind == "diagonal_affine":
        return X * model["slope"] + model["intercept"]
    if kind == "speaker_diagonal_affine":
        if speakers is None or len(speakers) != X.shape[0]:
            raise ValueError("one speaker key per prediction row is required")
        speakers = np.asarray(speakers).astype(str)
        prediction = X * model["slope"] + model["intercept"]
        for key, parameters in model["speaker_parameters"].items():
            mask = speakers == key
            prediction[mask] = X[mask] * parameters["slope"] + parameters["intercept"]
        return prediction
    if kind in ("pcr", "ridge"):
        latent = ((X - model["x_mean"]) / model["x_scale"]) @ model["components"].T
        return (latent @ model["latent_coef"]) * model["y_scale"] + model["y_mean"]
    if kind == "full_ridge":
        return (((X - model["x_mean"]) / model["x_scale"]) @ model["coef"]) * model["y_scale"] + model["y_mean"]
    raise ValueError(f"unknown converter: {kind}")


def _quantiles(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return {key: None for key in ("min", "p05", "median", "p95", "max")}
    return dict(zip(("min", "p05", "median", "p95", "max"),
                    np.quantile(finite, [0, .05, .5, .95, 1]).tolist()))


def compute_metrics(Y, Yhat, groups=None, target_scale=None,
                    return_per_dimension=False, wasserstein=False):
    """Metrics retain nonfinite row counts; no sample silently disappears.

    Scalar element errors cover jointly finite elements; row cosine covers
    wholly finite nonzero row pairs. Column correlation uses wholly finite
    rows and is undefined for constant columns. Summary reports these counts.
    With return_per_dimension=True returns (JSON-safe summary, array dictionary).
    target_scale must be fitted training-target standard deviations, never
    inferred from held-out values for NRMSE.
    """
    Y, Yhat = np.asarray(Y, dtype=np.float64), np.asarray(Yhat, dtype=np.float64)
    if Y.ndim != 2 or Y.shape != Yhat.shape or not Y.size:
        raise ValueError("metrics require equal nonempty 2D shapes")
    finite = np.isfinite(Y) & np.isfinite(Yhat)
    rows = finite.all(axis=1)
    error = np.full(Y.shape, np.nan)
    np.subtract(Yhat, Y, out=error, where=finite)
    errors = error[finite]
    maxabs = float(np.max(np.abs(errors))) if errors.size else None
    mae = float(np.mean(np.abs(errors))) if errors.size else None
    rmse = float(np.sqrt(np.mean(errors ** 2))) if errors.size else None
    a, b = Y[rows], Yhat[rows]
    norms_a, norms_b = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    cosine_valid = (norms_a > 0) & (norms_b > 0)
    cosines = np.full(len(a), np.nan)
    cosines[cosine_valid] = np.sum(a[cosine_valid] * b[cosine_valid], axis=1) / (
        norms_a[cosine_valid] * norms_b[cosine_valid])
    # Roundoff can put an exact-match cosine just outside [-1, 1].
    cosines[cosine_valid] = np.clip(cosines[cosine_valid], -1, 1)
    count = finite.sum(axis=0)
    column_mae = np.divide(np.nansum(np.abs(error), axis=0), count,
                           out=np.full(Y.shape[1], np.nan), where=count > 0)
    column_rmse = np.sqrt(np.divide(np.nansum(error ** 2, axis=0), count,
                                    out=np.full(Y.shape[1], np.nan), where=count > 0))
    correlations = np.full(Y.shape[1], np.nan)
    mean_difference = np.full(Y.shape[1], np.nan)
    std_difference = np.full(Y.shape[1], np.nan)
    if len(a):
        ac, bc = a - a.mean(axis=0), b - b.mean(axis=0)
        denominator = np.linalg.norm(ac, axis=0) * np.linalg.norm(bc, axis=0)
        np.divide(np.sum(ac * bc, axis=0), denominator, out=correlations,
                  where=denominator > 0)
        correlations = np.clip(correlations, -1, 1)
        mean_difference = b.mean(axis=0) - a.mean(axis=0)
        std_difference = b.std(axis=0) - a.std(axis=0)
    arrays = {"column_mae": column_mae, "column_rmse": column_rmse,
              "column_correlation": correlations,
              "column_mean_difference": mean_difference,
              "column_std_difference": std_difference}
    summary = {"shape": list(Y.shape), "total_scalar_elements": int(Y.size),
               "jointly_finite_elements": int(finite.sum()),
               "nonfinite_target_elements": int((~np.isfinite(Y)).sum()),
               "nonfinite_prediction_elements": int((~np.isfinite(Yhat)).sum()),
               "fully_finite_rows": int(rows.sum()),
               "nonfinite_rows": int((~rows).sum()),
               "max_absolute_error": maxabs, "mean_absolute_error": mae,
               "rmse": rmse, "cosine_valid_rows": int(cosine_valid.sum()),
               "cosine_undefined_finite_rows": int((~cosine_valid).sum()),
               "zero_target_rows": int((norms_a == 0).sum()),
               "zero_prediction_rows": int((norms_b == 0).sum()),
               "mean_cosine_similarity": float(np.mean(cosines[cosine_valid])) if cosine_valid.any() else None,
               "min_cosine_similarity": float(np.min(cosines[cosine_valid])) if cosine_valid.any() else None,
               "column_correlation_defined": int(np.isfinite(correlations).sum()),
               "column_correlation_undefined": int((~np.isfinite(correlations)).sum()),
               "column_correlation_quantiles": _quantiles(correlations),
               "column_mae_quantiles": _quantiles(column_mae),
               "column_rmse_quantiles": _quantiles(column_rmse),
               "absolute_column_mean_difference_quantiles": _quantiles(np.abs(mean_difference)),
               "absolute_column_std_difference_quantiles": _quantiles(np.abs(std_difference)),
               "nonfinite_policy": "count all; finite element errors; wholly finite rows for correlations/distributions/cosine"}
    if target_scale is not None:
        scale = np.asarray(target_scale, dtype=float)
        if scale.shape != (Y.shape[1],) or not np.isfinite(scale).all() or (scale < 0).any():
            raise ValueError("target_scale must be nonnegative finite training std per column")
        normalized_rmse = np.divide(column_rmse, scale, out=np.full_like(scale, np.nan),
                                    where=scale > 0)
        arrays["column_train_scale_nrmse"] = normalized_rmse
        summary["train_scale_nrmse_quantiles"] = _quantiles(normalized_rmse)
        summary["train_scale_nrmse_mean"] = float(np.nanmean(normalized_rmse)) if np.isfinite(normalized_rmse).any() else None
        summary["train_scale_zero_columns"] = int((scale == 0).sum())
        valid_scale = scale > 0
        normalized_square_sum = np.sum(np.nansum(error[:, valid_scale] ** 2, axis=0) /
                                       scale[valid_scale] ** 2)
        normalized_count = int(finite[:, valid_scale].sum())
        summary["global_train_scale_nrmse"] = float(np.sqrt(normalized_square_sum / normalized_count)) if normalized_count else None
        summary["global_train_scale_nrmse_definition"] = "sqrt(mean((prediction-target)^2 / supplied_train_scale^2)) over jointly finite entries with positive scale"
    if wasserstein:
        from scipy.stats import wasserstein_distance
        distances = np.array([wasserstein_distance(a[:, j], b[:, j])
                              for j in range(Y.shape[1])]) if len(a) else np.full(Y.shape[1], np.nan)
        arrays["column_wasserstein"] = distances
        summary["column_wasserstein_quantiles"] = _quantiles(distances)
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(Y):
            raise ValueError("groups length differs from metrics rows")
        group_mae = []
        for key in np.unique(groups):
            block = error[groups == key]
            valid = np.isfinite(block)
            group_mae.append(float(np.abs(block[valid]).mean()) if valid.any() else np.nan)
        arrays["dialogue_mae"] = np.asarray(group_mae)
        summary["dialogue_count"] = len(group_mae)
        summary["dialogue_equal_weight_mae"] = float(np.nanmean(group_mae)) if np.isfinite(group_mae).any() else None
        if target_scale is not None:
            group_nrmse = []
            for key in np.unique(groups):
                block = error[groups == key][:, valid_scale] / scale[valid_scale]
                valid = np.isfinite(block)
                group_nrmse.append(float(np.sqrt(np.mean(block[valid] ** 2))) if valid.any() else np.nan)
            arrays["dialogue_nrmse"] = np.asarray(group_nrmse)
            summary["dialogue_equal_nrmse"] = float(np.nanmean(group_nrmse)) if np.isfinite(group_nrmse).any() else None
            summary["dialogue_equal_nrmse_definition"] = "arithmetic mean of per-dialogue sqrt(mean((error / supplied_train_scale)^2)); dialogues equally weighted"
    return (summary, arrays) if return_per_dimension else summary
