#!/usr/bin/env python

import math
import numpy as np
import random

from tqdm import tqdm

from grouse import Grouse
from petrels import Petrels


class GrouseProjection:

    def __init__(self, n_components, step_size=1, random_state=None):

        self.n_components = n_components
        random.seed(random_state)

        self.step_size = step_size

    def fit_transform(self, X):

        tracker = Grouse(X.shape[1], self.n_components, step=self.step_size)

        Y = np.zeros((X.shape[0], self.n_components))

        for i in tqdm(range(X.shape[0]), desc="Projecting X via Grouse..."):
            x = X[i, :].reshape(-1, 1)

            ss = np.ones_like(x)
            tracker.consume(x, ss)

            proj = tracker._project(x, ss)

            Y[i, :] = proj.T

        return Y


    def transform(self, X):
        return self.fit_transform(X)

