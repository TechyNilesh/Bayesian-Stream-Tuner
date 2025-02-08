from capymoa.base import Regressor
from typing import Dict, Tuple
from scipy.stats import norm
from capymoa.instance import RegressionInstance
from capymoa.stream import Schema
import numpy as np


class BayesianLinearRegression(Regressor):

    def __init__(self, schema: Schema = None, random_seed=1, alpha=1, beta=1, smoothing=None):
        super().__init__(schema=schema, random_seed=random_seed)
        self.alpha = alpha  # Prior parameter
        self.beta = beta    # Noise parameter
        self.smoothing = smoothing  # Smoothing factor for concept drift
        # Covariance matrix (Σ)
        self._ss: Dict[Tuple[int, int], float] = {}
        # Inverse covariance matrix (Σ^{-1})
        self._ss_inv: Dict[Tuple[int, int], float] = {}
        self._m: Dict[int, float] = {}    # Mean vector (μ)
        self._n_features = None  # Number of features, to be determined during training

    def _get_arrays(self, features):
        """Retrieve the parameter arrays for the given features."""
        n = len(features)
        m_arr = np.array([self._m.get(f, 0.0) for f in features])
        ss_arr = np.array([
            [self._ss.get(
                min((features[i], features[j]), (features[j], features[i])),
                1.0 / self.alpha if features[i] == features[j] else 0.0)
             for j in range(n)]
            for i in range(n)
        ])
        ss_inv_arr = np.array([
            [self._ss_inv.get(
                min((features[i], features[j]), (features[j], features[i])),
                self.alpha if features[i] == features[j] else 0.0)
             for j in range(n)]
            for i in range(n)
        ], order='F')
        return m_arr, ss_arr, ss_inv_arr

    def _set_arrays(self, features, m_arr, ss_arr, ss_inv_arr):
        """Set the parameter arrays for the given features."""
        n = len(features)
        for i in range(n):
            f_i = features[i]
            self._m[f_i] = m_arr[i]
            ss_row = ss_arr[i]
            ss_inv_row = ss_inv_arr[i]
            for j in range(n):
                f_j = features[j]
                key = min((f_i, f_j), (f_j, f_i))
                self._ss[key] = ss_row[j]
                self._ss_inv[key] = ss_inv_row[j]

    def train(self, instance: RegressionInstance):
        x_dict = {i: value for i, value in enumerate(instance.x)}
        y = instance.y_value

        if self._n_features is None:
            self._n_features = self.schema.get_num_attributes()

        features = list(x_dict.keys())
        x_arr = np.array([x_dict[f] for f in features])

        m_arr, ss_arr, ss_inv_arr = self._get_arrays(features)
        bx = self.beta * x_arr

        if self.smoothing is None:
            # Sherman-Morrison formula to update inverse covariance matrix
            v1 = ss_inv_arr @ bx
            denom = 1 + (v1 @ x_arr)
            ss_inv_arr -= np.outer(v1, v1) / denom

            # Update mean vector (Equation 3.50 from Bishop)
            m_arr = ss_inv_arr @ (ss_arr @ m_arr + bx * y)

            # Update covariance matrix (Equation 3.51 from Bishop)
            ss_arr += np.outer(bx, x_arr)
        else:
            # Apply smoothing for concept drift adaptation
            new_ss_arr = self.smoothing * ss_arr + \
                (1 - self.smoothing) * np.outer(bx, x_arr)
            ss_inv_arr = np.linalg.inv(new_ss_arr)
            m_arr = ss_inv_arr @ (self.smoothing * ss_arr @
                                  m_arr + (1 - self.smoothing) * bx * y)
            ss_arr = new_ss_arr

        # Update the internal state with new parameters
        self._set_arrays(features, m_arr, ss_arr, ss_inv_arr)

    def predict(self, instance: RegressionInstance, with_dist=False):
        x_dict = {i: value for i, value in enumerate(instance.x)}
        features = list(x_dict.keys())
        x_arr = np.array([x_dict[f] for f in features])

        m_arr, _, ss_inv_arr = self._get_arrays(features)

        # Compute the predictive mean (Equation 3.58 from Bishop)
        y_pred_mean = m_arr @ x_arr

        if not with_dist:
            return y_pred_mean

        # Compute the predictive variance (Equation 3.59 from Bishop)
        y_pred_var = 1 / self.beta + x_arr @ ss_inv_arr @ x_arr
        y_pred_std = np.sqrt(y_pred_var)

        return y_pred_mean, y_pred_std

    def predict_proba(self, instance: RegressionInstance):
        mean, std = self.predict(instance, with_dist=True)
        return norm.pdf(instance.y_value, loc=mean, scale=std)

    def __str__(self):
        return "BayesianLinearRegression"