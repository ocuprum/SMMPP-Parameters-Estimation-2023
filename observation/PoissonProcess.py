import numpy as np

rng = np.random.default_rng()

class PoissonProcess:
    def __init__(self, lam):
        self.lam = lam


    def next(self):
        return rng.poisson(self.lam)