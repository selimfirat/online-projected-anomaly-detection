#!/usr/bin/env python

import itertools
import numpy as np
import tqdm
tqdm.tqdm.monitor_interval = 0

class Chain:

    def __init__(self, deltamax, depth=25):
        k = len(deltamax)
        self.deltamax = deltamax # feature ranges
        self.depth = depth
        self.fs = [np.random.randint(0, k) for d in range(depth)]
        self.cmsketches = [{} for i in range(depth)] * depth
        self.cmsketches_cur = [{} for i in range(depth)] * depth
        self.shift = np.random.rand(k) * deltamax

        self.is_first_window = True

        self.alpha = 1.006 # 1.004

        self.t = 1

    def fit(self, X, update=False):
        prebins = np.zeros(X.shape, dtype=np.float)
        depthcount = np.zeros(len(self.deltamax), dtype=np.int)
        for depth in range(self.depth):
            f = self.fs[depth]
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[:,f] = (X[:,f] + self.shift[f])/self.deltamax[f]
            else:
                prebins[:,f] = 2.0*prebins[:,f] - self.shift[f]/self.deltamax[f]

            if self.is_first_window:
                cmsketch = self.cmsketches[depth]
                for prebin in prebins:
                    l = tuple(np.floor(prebin).astype(np.int))
                    if not l in cmsketch:
                        cmsketch[l] = 0
                    cmsketch[l] += 1 * self.factor() ##

                self.cmsketches[depth] = cmsketch

                self.cmsketches_cur[depth] = cmsketch

            else:
                cmsketch = self.cmsketches_cur[depth]

                for prebin in prebins:
                    l = tuple(np.floor(prebin).astype(np.int))
                    if not l in cmsketch:
                        cmsketch[l] = 0
                    cmsketch[l] += 1 * self.factor() ##

                self.cmsketches_cur[depth] = cmsketch

        self.t += 1 ##

        return self

    def partial_fit(self, X):
        return self.fit(X, update=True)

    def bincount(self, X):
        scores = np.zeros((X.shape[0], self.depth))
        prebins = np.zeros(X.shape, dtype=np.float)
        depthcount = np.zeros(len(self.deltamax), dtype=np.int)
        for depth in range(self.depth):
            f = self.fs[depth]
            depthcount[f] += 1

            if depthcount[f] == 1:
                prebins[:,f] = (X[:,f] + self.shift[f])/self.deltamax[f]
            else:
                prebins[:,f] = 2.0*prebins[:,f] - self.shift[f]/self.deltamax[f]

            cmsketch = self.cmsketches[depth]
            for i, prebin in enumerate(prebins):
                l = tuple(np.floor(prebin).astype(np.int))
                if not l in cmsketch:
                    scores[i,depth] = 0.0
                else:
                    scores[i,depth] = cmsketch[l]

        return scores

    def factor(self):
        return np.power(self.alpha, np.sqrt(self.t))

    def score(self, X):
        # scale score logarithmically to avoid overflow:
        #    score = min_d [ log2(bincount x 2^d) = log2(bincount) + d ]
        scores = self.bincount(X)
        depths = np.array([d for d in range(1, self.depth+1)])
        scores = np.log2(1.0 + scores) + depths # add 1 to avoid log(0)
        return int(np.min(scores, axis=1) * 1.0 / self.factor())

    def next_window(self):
        self.is_first_window = False
        self.cmsketches = self.cmsketches_cur
        self.cmsketches_cur = [{} for i in range(self.depth)] * self.depth


class HSChains:
    def __init__(self, k, nchains=100, depth=25, seed=42):
        self.nchains = nchains
        self.depth = depth
        self.chains = []

        for i in range(self.nchains):
            deltamax = 0.5 * np.ones((k,))

            c = Chain(deltamax, depth=self.depth)
            self.chains.append(c)

    def score(self, X):
        scores = np.zeros(X.shape[0])
        for ch in self.chains:
            scores += ch.score(X)

        scores /= float(self.nchains)
        return scores

    def partial_fit(self, X):
        for ch in self.chains:
            ch.partial_fit(X)

    def next_window(self):
        for ch in self.chains:
            ch.next_window()


