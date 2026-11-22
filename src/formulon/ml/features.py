"""Physics-derived feature engineering (scikit-learn style, no sklearn needed)."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .vectorize import vectorize

__all__ = ["FormulaFeatures"]


class FormulaFeatures:
    """Append physically meaningful columns computed by Formulon formulas.

    Example
    -------
    >>> ff = FormulaFeatures({
    ...     "re": ("reynolds_number", dict(density="rho", flow_speed="v",
    ...                                    characteristic_length="L", viscosity="mu")),
    ... })
    >>> X = {"rho": [1000.], "v": [2.], "L": [0.05], "mu": [1e-3]}
    >>> ff.fit_transform(X)["re"]
    array([100000.])

    Inputs may be a dict of columns or a pandas DataFrame (anything with
    ``__getitem__`` by column name).  Output is a dict of NumPy columns;
    ``to_matrix`` stacks them for model input.  Out-of-domain rows become NaN
    (``on_invalid='mask'``) so a pipeline can drop or impute them explicitly.
    """

    def __init__(self, specs: Mapping[str, tuple], on_invalid: str = "mask"):
        import formulon

        self.specs = dict(specs)
        self._fns = {}
        for feat, (fname, _mapping) in self.specs.items():
            fn = getattr(formulon, fname, None)
            if fn is None:
                raise KeyError(f"unknown formula {fname!r} for feature {feat!r}")
            self._fns[feat] = vectorize(fn, on_invalid=on_invalid)
        self.feature_names_ = list(self.specs)
        self.invalid_fraction_ = {}

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X) -> dict:
        out = {}
        for feat, (_fname, mapping) in self.specs.items():
            kwargs = {arg: np.asarray(X[col], dtype=float) for arg, col in mapping.items()}
            out[feat] = self._fns[feat](**kwargs)
            rep = self._fns[feat].report
            self.invalid_fraction_[feat] = 0.0 if rep is None else 1.0 - rep.valid_fraction
        return out

    def fit_transform(self, X, y=None) -> dict:
        return self.fit(X).transform(X)

    def to_matrix(self, features: Mapping[str, np.ndarray], order: Sequence[str] | None = None) -> np.ndarray:
        order = list(order or self.feature_names_)
        return np.column_stack([np.asarray(features[k], dtype=float) for k in order])
