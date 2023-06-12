import numpy as np
import scipy.stats as stats 

rng = np.random.default_rng()

class HMM:
    def __init__(self, lams, max_soj_time, obs_max): 
        self.states_amount = lams.size * max_soj_time       
        self.max_soj_time = max_soj_time
        self.lams = lams
        self.obs_max = obs_max

        normalize = lambda arr: arr / arr.sum()
        delta = 1 / lams.size / 10
        low, high = 1 / lams.size - delta, 1 / lams.size + delta

        self.__gen_initial(normalize, low, high)
        self.__gen_trans(normalize, low, high)
        self.compute_out()     


    def __gen_initial(self, normalize, low, high):
        self.initial = np.zeros(self.states_amount)
        self.initial[0::self.max_soj_time] = rng.uniform(low, high, size=self.lams.size)
        self.initial = normalize(self.initial)


    def __gen_trans(self, normalize, low, high):
        self.trans = np.zeros((self.states_amount, self.states_amount))
        for ind1 in range(self.states_amount):
            if ind1 % self.max_soj_time != self.max_soj_time - 1:
                ind2 = ind1 + 1
                self.trans[ind1, ind2] = rng.uniform(low, high)

            for ind2 in range(0, self.states_amount, self.max_soj_time):
                if ind1 != ind2 and ind1 // self.max_soj_time != ind2 // self.max_soj_time:
                    self.trans[ind1, ind2] = rng.uniform(low, high)
        self.trans = np.apply_along_axis(normalize, 1, self.trans) 


    def compute_out(self):
        self.out = np.zeros((self.states_amount, self.obs_max+1))
        for ind in range(self.states_amount):
            lam = self.lams[ind // self.max_soj_time]
            for y in range(self.obs_max+1):
                self.out[ind, y] = stats.poisson.pmf(k=y, mu=lam) 


    def __repr__(self) -> str:
        indist = '\nINITIAL DISTRIBUTION: \n{}\n'.format(self.initial.round(4))
        tm = '\nTRANSITION MATRIX: \n{}\n'.format(self.trans.round(4))
        lams = '\nLAMS: \n{}\n\n'.format(self.lams)

        return indist + tm + lams
    

    def copy(self):
        hmm_copy = HMM(lams=self.lams, max_soj_time=self.max_soj_time, obs_max=self.obs_max)
        hmm_copy.initial = np.copy(self.initial)
        hmm_copy.trans = np.copy(self.trans)
        hmm_copy.out = np.copy(self.out)
        
        return hmm_copy