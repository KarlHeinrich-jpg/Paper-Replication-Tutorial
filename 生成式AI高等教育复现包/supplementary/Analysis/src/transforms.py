from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class UseTerms(BaseEstimator, TransformerMixin):
    def __init__(self, enabled=False):
        self.enabled = enabled

    @staticmethod
    def _validate(X):
        if not isinstance(X, pd.DataFrame) or not {"literacy", "use"}.issubset(X.columns):
            raise ValueError("UseTerms requires a DataFrame with literacy and use.")
        z = X[["literacy", "use"]].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(z.to_numpy(float)).all():
            raise ValueError("Focal composites cannot be missing/nonfinite before nonlinear-term construction.")
        return z

    def fit(self, X, y=None):
        z = self._validate(X)
        if len(z) == 0:
            raise ValueError("Empty training partition.")
        self.L_mean_ = float(z.literacy.mean())
        self.U_mean_ = float(z.use.mean())
        self.feature_names_in_ = np.asarray(X.columns, object)
        self.n_features_in_ = len(X.columns)
        if X.isna().all(axis=0).any():
            raise ValueError("A training predictor is entirely missing; the declared feature set is not estimable.")
        return self

    def transform(self, X):
        check_is_fitted(self, ["L_mean_", "U_mean_"])
        z = self._validate(X)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Prediction columns/order do not match the fitted pipeline.")
        d = X.copy()
        d[["literacy", "use"]] = z
        if self.enabled:
            d["use_sq"] = (z.use - self.U_mean_) ** 2
            d["literacy_use"] = (z.literacy - self.L_mean_) * (z.use - self.U_mean_)
            d["literacy"] = z.literacy - self.L_mean_
            d["use"] = z.use - self.U_mean_
        return d
