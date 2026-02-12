from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.linalg import eigh
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from xdflow.transforms.sklearn_transform import SKLearnTransformer


class CholeskyLDA(BaseEstimator, ClassifierMixin):
    """
    Fast LDA using a Cholesky factorization of the (shrunk) pooled within-class
    covariance (feature-space 'primal') + a small CxC eigendecomposition for .transform.

    Behavior for covariance shrinkage:
      - If `covariance_estimator` is provided (e.g., sklearn.covariance.OAS()):
          * If cov_estimator_on_within=True (default): clone and fit it on within-class
            residuals Xc.
          * If cov_estimator_on_within=False: clone and fit it on raw X.
          Its `covariance_` is used directly as Σ.
      - Else if `shrinkage` is a float in [0,1], use Σ = (1-α) S + α μ I,
        where S is the pooled within-class covariance and μ = tr(S)/p.
      - Else if `shrinkage is None`, Σ = S (no shrinkage).

    Parameters
    ----------
    shrinkage : float in [0, 1] or None, default=None
        If float, use Σ = (1-α) S + α μ I with α=shrinkage and μ=tr(S)/p.
        If None, use Σ = S (no shrinkage). Ignored if `covariance_estimator` is set.

    covariance_estimator : estimator or None, default=None
        An sklearn-style covariance estimator (e.g., sklearn.covariance.OAS()).
        It is cloned and fit on either within-class residuals or raw X depending
        on `cov_estimator_on_within`.

    cov_estimator_on_within : bool, default=True
        If True, fit the covariance estimator on within-class residuals Xc and set
        `assume_centered=True` when supported. If False, fit on raw X and do not
        modify `assume_centered`.

    cov_estimator_per_class : bool, default=False
        If True, fit the covariance estimator separately for each class and mix with
        priors (sklearn solver="lsqr" behavior). If False, use original single-fit approach.

    priors : array-like of shape (n_classes,), default=None
        Class prior probabilities. If None, inferred from data.

    dtype : {"float32","float64"}, default="float32"
        Internal compute dtype for heavy ops. Outputs are float64.

    store_covariance : bool, default=False
        If True, stores diag(Σ) in the private `_covariance_diag_`.

    Attributes (after fit)
    ----------------------
    classes_ : (C,)
    priors_  : (C,)
    means_   : (C, p)
    coef_    : (C, p)          # rows are Σ^{-1} μ_k
    intercept_ : (C,)          # -0.5 μ_k^T Σ^{-1} μ_k + log π_k
    scalings_ : (p, r)         # projection for transform (r = C-1)
    explained_variance_ratio_ : (r,)
    xbar_    : (p,)            # prior-weighted global mean used for transform centering

    Diagnostics
    -----------
    shrinkage_ : float or None
        Shrinkage alpha actually used, when available (from estimator or float path).
    mu_ : float or None
        Average variance μ = tr(S)/p (if computed on the float/none paths).
    """

    def __init__(
        self,
        shrinkage: float | None = None,
        covariance_estimator: Any | None = None,
        cov_estimator_on_within: bool = True,
        cov_estimator_per_class: bool = False,
        priors: Any | None = None,
        dtype: str = "float32",
        store_covariance: bool = False,
    ):
        self.shrinkage = shrinkage
        self.covariance_estimator = covariance_estimator
        self.cov_estimator_on_within = cov_estimator_on_within
        self.cov_estimator_per_class = cov_estimator_per_class
        self.priors = priors
        self.dtype = dtype
        self.store_covariance = store_covariance

        # private diagnostics
        self._alpha = None
        self._mu_scalar = None
        self._N_eff = None
        self._covariance_diag_ = None

    # ----------------- helpers -----------------

    @staticmethod
    def _class_stats(X, y, classes):
        means = np.vstack([X[y == c].mean(axis=0) for c in classes])
        n_k = np.array([(y == c).sum() for c in classes], dtype=np.int64)
        priors = n_k / n_k.sum()
        return means, n_k, priors

    # ----------------- sklearn API -----------------

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, ensure_2d=True)
        dt = np.float32 if str(self.dtype) == "float32" else np.float64
        X = np.asarray(X, dtype=dt, order="C")
        y = np.asarray(y)

        classes = unique_labels(y)
        C = classes.size
        if C < 2:
            raise ValueError("LDA requires at least two classes.")

        n, p = X.shape
        means, n_k, priors_emp = self._class_stats(X, y, classes)

        # priors
        if self.priors is None:
            priors = priors_emp.astype(dt)
        else:
            priors = np.asarray(self.priors, dtype=dt)
            if priors.shape != (C,):
                raise ValueError(f"`priors` must have shape ({C},), got {priors.shape}.")
            if np.any(priors < 0):
                raise ValueError("`priors` must be non-negative.")
            s = float(priors.sum())
            priors = priors / (s if s > 0 else 1.0)

        # pooled within-class residuals (used for Σ path and later steps)
        y_inv = np.searchsorted(classes, y)
        Xc = X - means[y_inv]
        N_eff = int(n - C)
        if N_eff <= 0:
            raise ValueError("Need N_eff = n - C > 0 (at least one class with ≥2 samples).")

        # ---- Build Σ (final covariance to factor) ----
        used_alpha = None
        used_mu = None

        if self.covariance_estimator is not None:
            if self.cov_estimator_per_class:
                # Fit the estimator per class, then mix with priors
                covs = []
                for _k, c in enumerate(classes):
                    if self.cov_estimator_on_within:
                        Xk = Xc[y == c]
                    else:
                        Xk = X[y == c]
                    est_k = clone(self.covariance_estimator)
                    # Do NOT override assume_centered; use whatever the user passed.
                    est_k.fit(Xk)
                    if not hasattr(est_k, "covariance_"):
                        raise ValueError("Provided covariance_estimator lacks `covariance_` after fit().")
                    Ck = np.asarray(est_k.covariance_, dtype=dt)
                    if Ck.shape != (p, p):
                        raise ValueError(f"covariance_estimator.covariance_ has shape {Ck.shape}, expected ({p},{p}).")
                    covs.append(Ck)

                # Weight by the SAME priors you use in the classifier (self.priors or empirical)
                Sigma = np.zeros((p, p), dtype=dt)
                for wk, Ck in zip(priors, covs):
                    Sigma += wk * Ck

                # Record shrinkage if exposed (e.g., OAS.shrinkage_) from first fitted estimator
                if len(covs) > 0:
                    # Check the first fitted estimator for shrinkage info
                    temp_est = clone(self.covariance_estimator)
                    if self.cov_estimator_on_within:
                        temp_est.fit(Xc[y == classes[0]])
                    else:
                        temp_est.fit(X[y == classes[0]])

                    if hasattr(temp_est, "shrinkage_"):
                        try:
                            used_alpha = float(np.asarray(temp_est.shrinkage_))
                        except Exception:
                            used_alpha = None
                used_mu = float(np.trace(Sigma) / p)
            else:
                # Original single-fit approach
                est = clone(self.covariance_estimator)

                if self.cov_estimator_on_within:
                    est.fit(Xc)
                else:
                    est.fit(X)

                if not hasattr(est, "covariance_"):
                    raise ValueError("Provided covariance_estimator lacks `covariance_` after fit().")

                Sigma = np.asarray(est.covariance_, dtype=dt)
                if Sigma.shape != (p, p):
                    raise ValueError(f"covariance_estimator.covariance_ has shape {Sigma.shape}, expected ({p},{p}).")
                # Record shrinkage if exposed (e.g., OAS.shrinkage_)
                if hasattr(est, "shrinkage_"):
                    try:
                        used_alpha = float(np.asarray(est.shrinkage_))
                    except Exception:
                        used_alpha = None
                used_mu = float(np.trace(Sigma) / p)

        else:
            # keep your no-estimator path; with sklearn parity it should use / n
            S = (Xc.T @ Xc) / dt(n)  # (p, p)
            mu = float(np.trace(S) / p)

            if self.shrinkage is None:
                Sigma = S
                used_alpha = 0.0
                used_mu = mu
            else:
                alpha = float(self.shrinkage)
                if not (0.0 <= alpha <= 1.0):
                    raise ValueError("`shrinkage` float must be in [0, 1].")
                Sigma = (dt(1.0) - dt(alpha)) * S
                Sigma.flat[:: p + 1] += dt(alpha * mu)
                used_alpha = alpha
                used_mu = mu

        # Enforce symmetry (numerical safety)
        Sigma = 0.5 * (Sigma + Sigma.T)

        # ---- Factor Σ and compute classifier + transform ----
        cho = cho_factor(Sigma, overwrite_a=False, check_finite=False)

        # coef_ = Σ^{-1} μ_k
        coef = cho_solve(cho, means.T, check_finite=False).T  # (C, p)
        quad = np.einsum("ij,ij->i", means, coef)  # (C,)
        intercept = -0.5 * quad + np.log(np.clip(priors, 1e-12, 1.0)).astype(dt)

        # transform: K = B^T Σ^{-1} B, with B = [sqrt(n_k/n) (μ_k - xbar)]
        xbar = np.average(means, axis=0, weights=priors)  # (p,)
        B = (means - xbar) * np.sqrt(n_k / dt(n))[:, None]  # (C, p)
        B = B.T  # (p, C)
        SigInvB = cho_solve(cho, B, check_finite=False)  # (p, C)
        K = B.T @ SigInvB  # (C, C)

        evals, V = eigh(K)  # ascending
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        V = V[:, order]

        r = C - 1
        if r > 0:
            scalings = SigInvB @ V[:, :r]  # (p, r)
            norms = np.linalg.norm(scalings, axis=0)
            norms[norms == 0] = 1.0
            scalings = scalings / norms
            pos = np.clip(evals[:r], 0.0, None)
            explained = (pos / (pos.sum() + 1e-12)).astype(np.float64)
        else:
            scalings = np.zeros((p, 0), dtype=dt)
            explained = np.zeros((0,), dtype=np.float64)

        # ---- store public attrs ----
        self.classes_ = classes
        self.priors_ = priors.astype(np.float64)
        self.means_ = means.astype(np.float64)
        self.coef_ = coef.astype(np.float64)
        self.intercept_ = intercept.astype(np.float64)
        self.scalings_ = scalings.astype(np.float64)
        self.explained_variance_ratio_ = explained
        self.xbar_ = xbar.astype(np.float64)

        # ---- private diagnostics ----
        self._alpha = used_alpha
        self._mu_scalar = used_mu
        self._N_eff = N_eff
        if self.store_covariance:
            self._covariance_diag_ = np.diag(Sigma).astype(np.float64)

        return self

    def decision_function(self, X):
        check_is_fitted(self, attributes=["coef_", "intercept_", "classes_"])
        X = check_array(X, accept_sparse=False, ensure_2d=True)
        return X @ self.coef_.T + self.intercept_

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        scores = self.decision_function(X)
        logZ = logsumexp(scores, axis=1, keepdims=True)
        return np.exp(scores - logZ)

    def transform(self, X):
        check_is_fitted(self, attributes=["scalings_", "xbar_"])
        X = check_array(X, accept_sparse=False, ensure_2d=True)
        if self.scalings_.shape[1] == 0:
            return np.zeros((X.shape[0], 0), dtype=X.dtype)
        return (X - self.xbar_) @ self.scalings_

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    # Diagnostics (read-only)
    @property
    def shrinkage_(self):
        check_is_fitted(self, attributes=["classes_"])
        return self._alpha

    @property
    def mu_(self):
        check_is_fitted(self, attributes=["classes_"])
        return self._mu_scalar


class CholeskyLDATransformer(SKLearnTransformer):
    """
    SKLearnTransformer wrapper for CholeskyLDA.

    Parameters
    ----------
    sample_dim : str
        The name of the dimension that corresponds to samples.
    target_coord : str, optional
        Coordinate containing the target variable for supervised fitting. Default "stimulus".
    output_dim_name : str, optional
        Name for the new dimension created by the transformer. Default "component".
    shrinkage : float in [0, 1] or None, default=None
        Shrinkage to use for LDA. Ignored if `covariance_estimator` is provided.
    covariance_estimator : object, optional
        Any sklearn-compatible covariance estimator instance (e.g., `sklearn.covariance.OAS()`).
    cov_estimator_on_within : bool, default=True
        Whether to fit the covariance estimator on within-class residuals (True) or raw X (False).
    cov_estimator_per_class : bool, default=False
        If True, fit the covariance estimator separately for each class and mix with
        priors (sklearn solver="lsqr" behavior). If False, use original single-fit approach.
    priors : array-like of shape (n_classes,), default=None
        Class prior probabilities. If None, inferred from data.
    dtype : {"float32","float64"}, default="float32"
        Internal compute dtype for heavy ops. Outputs are float64.
    store_covariance : bool, default=False
        If True, stores diag(Σ) in the private `_covariance_diag_`.
    sel : dict, optional
        Selection criteria for input data.
    drop_sel : dict, optional
        Drop selection criteria for input data.
    """

    def __init__(
        self,
        sample_dim: str,
        target_coord: str = "stimulus",
        output_dim_name: str = "component",
        shrinkage: float | None = None,
        covariance_estimator: object | None = None,
        cov_estimator_on_within: bool = True,
        cov_estimator_per_class: bool = False,
        priors: Iterable[tuple[Any, Any]] | None = None,
        dtype: str = "float32",
        store_covariance: bool = False,
        sel=None,
        drop_sel=None,
    ):
        super().__init__(
            estimator_cls=CholeskyLDA,
            sample_dim=sample_dim,
            target_coord=target_coord,
            output_dim_name=output_dim_name,
            shrinkage=shrinkage,
            covariance_estimator=covariance_estimator,
            cov_estimator_on_within=cov_estimator_on_within,
            cov_estimator_per_class=cov_estimator_per_class,
            priors=priors,
            dtype=dtype,
            store_covariance=store_covariance,
            sel=sel,
            drop_sel=drop_sel,
        )
