#!/usr/bin/env python

import math
import numpy as np
import random

from tqdm import tqdm

from petrels import Petrels


class PETRELSProjection:

    def __init__(self, n_components, random_state=None):

        self.n_components = n_components
        random.seed(random_state)

    def fit_transform(self, X):
        tracker = Petrels(X.shape[1], self.n_components, .98)

        Y = np.zeros((X.shape[0], self.n_components))

        for i in tqdm(range(X.shape[0])):
            x = X[i, :].reshape(-1, 1)

            ss = x # np.ones_like(x)
            tracker.consume(x, ss)

            proj = tracker._project(x, ss)

            Y[i, :] = proj.T

        return Y

    def transform(self, X):
        return self.fit_transform(X)

