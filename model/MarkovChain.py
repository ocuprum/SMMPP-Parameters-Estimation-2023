import numpy as np
from observation.SemiMarkovChain import get_prob_intervals, get_state

rng = np.random.default_rng()

class MarkovChain:
    def __init__(self, smc):
        self.max_soj_time = smc.kernel.shape[2]
        self.Z, self.D = smc.kernel.shape[0], self.max_soj_time-1
        self.states_amount = self.Z * self.max_soj_time

        self.__get_initial(smc)
        self.__get_soj_time(smc)
        self.__get_trans_mtrx(smc)

        u = rng.uniform(0, 1)
        self.U = get_state(u, self.initial_dist)


    def __get_initial(self, smc):
        self.initial_dist = np.zeros(self.states_amount)
        self.initial_cdf = np.zeros(self.states_amount)

        self.initial_dist[0::self.max_soj_time] = smc.initial_dist
        self.initial_cdf[0::self.max_soj_time] = smc.initial_cdf


    def __get_soj_time(self, smc):
        self.soj_time_dist = smc.kernel.sum(axis=1)
        self.soj_time_cdf = np.apply_along_axis(get_prob_intervals, 1, self.soj_time_dist)


    def __get_trans_mtrx(self, smc):
        self.trans_mtrx_dist = np.zeros((self.states_amount, self.states_amount))

        for ind1 in range(self.states_amount):
            i, d = ind1 // self.max_soj_time, ind1 % self.max_soj_time

            if d <= self.D:
                denom = (1-self.soj_time_cdf[i, d-1]) if d != 0 else 1
                if d <= self.D-1:
                    ind2 = ind1 + 1
                    self.trans_mtrx_dist[ind1, ind2] = (1-self.soj_time_cdf[i, d]) / denom

                for ind2 in range(0, self.states_amount, self.max_soj_time):
                    j = ind2 // self.max_soj_time
                    if ind1 != ind2 and i != j:
                        self.trans_mtrx_dist[ind1, ind2] = smc.kernel[i, j, d] / denom

        self.trans_mtrx_cdf = np.apply_along_axis(get_prob_intervals, 1, self.trans_mtrx_dist)


    def next(self):
        u = rng.uniform(0, 1)
        self.U = get_state(u, self.trans_mtrx_cdf[self.U])


    def __repr__(self):
        sa = '\nSTATES AMOUNT: {}\n'.format(self.states_amount)
        mst = '\nMAX SOJOURN TIME: {}\n'.format(self.max_soj_time)
        indist = '\nINITIAL DISTRIBUTION: \n{}\n'.format(self.initial_dist)
        tm = '\nTRANSITION MATRIX: \n{}\n'.format(self.trans_mtrx_dist.round(2))
        
        return sa + mst + indist + tm