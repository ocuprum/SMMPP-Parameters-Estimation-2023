import numpy as np

rng = np.random.default_rng()

class SemiMarkovChain:
    def __init__(self, max_time, initial_dist, kernel):
        self.initial_dist = initial_dist
        self.initial_cdf = get_prob_intervals(initial_dist)
        self.states_amount = initial_dist.shape[0]

        self.kernel = kernel
        self.__from_kernel()

        self.emb_trans_cdf = np.apply_along_axis(get_prob_intervals, 1, self.emb_trans)
        self.cond_soj_cdf = np.apply_along_axis(get_prob_intervals, 2, self.cond_soj_dist)

        self.max_time = max_time

        rand_u = rng.uniform(0, 1)
        self.X = get_state(rand_u, self.initial_cdf)
        self.S = 0
    

    def __from_kernel(self):
        '''Отримання матриці перехідних ймовірностей вкладеного ланцюга 
        та умовних розподілів часів перебування у станах'''

        self.emb_trans = self.kernel.sum(axis=2)

        repl_emb_trans = np.copy(self.emb_trans)
        repl_emb_trans[repl_emb_trans == 0] = 1

        self.cond_soj_dist = np.zeros(self.kernel.shape)
        for i in range(self.cond_soj_dist.shape[0]):
            for j in range(self.cond_soj_dist.shape[1]):
                self.cond_soj_dist[i, j] = self.kernel[i, j] / repl_emb_trans[i, j]


    def _restart(self):
        rand_u = rng.uniform(0, 1)
        self.X = get_state(rand_u, self.initial_cdf)
        self.S = 0


    def next(self):
        '''Генерування наступного стану ланцюга'''

        state_u = rng.uniform(0, 1)
        prev_X = self.X
        self.X = get_state(state_u, self.emb_trans_cdf[prev_X])

        soj_u = rng.uniform(0, 1)
        soj_time = get_state(soj_u, self.cond_soj_cdf[prev_X, self.X]) + 1
        self.S += soj_time

        if self.S >= self.max_time: return False

        return soj_time


# Визначення наступного стану ланцюга 
def get_state(u, cdf):
    for i in range(cdf.shape[0]):
        if u <= cdf[i]: return i


# Формування функції розподілу
def get_prob_intervals(arr):
    intervals = np.zeros(arr.shape[0])

    intervals[0] = arr[0]
    for i in range(1, arr.shape[0]):
        intervals[i] = intervals[i-1] + arr[i]
        if intervals[i] == 1: break
    
    return intervals