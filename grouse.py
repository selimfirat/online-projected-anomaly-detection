# Copyright 2015 Gregory Hasseler
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from tracker import Tracker


class Grouse(Tracker):
    """
    This class is an implementation of the Grassmannian Rank-One Update
    Subspace Estimation (GROUSE) subspace tracking algorithm, presented by
    Balzano et al. in  http://arxiv.org/abs/1006.4046

    This class assumes the use of Numpy vectors.
    """

    def __init__(self, ambient_dim, rank, step):
        """
        Keyword arguments:
        ambient_dim -- ambient dimension of the observations
        rank -- estimate of the rank
        step -- step size
        """
        self.ambient_dim = ambient_dim
        self.rank = rank
        self.step = step

        # Initialize our subspace estimate to a random orthogonal matrix
        self.U = np.linalg.qr(np.random.rand(ambient_dim, rank))[0]        

    def consume(self, ob_vec, sample_vec, step=None):
        if step is None:
            step = self.step

        # Find our projection into the subspace
        projection = self._project(ob_vec, sample_vec)

        # Find the imputed measurement residual
        row_indices = np.nonzero(sample_vec)[0]
        imputed_measurement = self.U @ projection
        residual = np.zeros((self.ambient_dim, 1))
        residual[row_indices, :] = ob_vec[row_indices, :] - imputed_measurement[row_indices, :]

        sigma = np.linalg.norm(residual) * np.linalg.norm(imputed_measurement)

        normalized_imputed_measurement = np.nan_to_num(imputed_measurement / np.linalg.norm(imputed_measurement))
        normalized_residual = np.nan_to_num(residual / np.linalg.norm(residual))
        normalized_projection = np.nan_to_num(projection.T / np.linalg.norm(projection))

        lhs_inner_update = (np.cos(sigma * step) - 1) * normalized_imputed_measurement
        rhs_inner_update = np.sin(sigma * step) * normalized_residual
        rhs = (lhs_inner_update + rhs_inner_update) @ normalized_projection

        self.U = self.U + rhs
