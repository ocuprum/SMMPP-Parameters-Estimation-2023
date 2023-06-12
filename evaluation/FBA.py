import numpy as np
from model.HMM import HMM


class FBA:
    '''Forward-backward algorithm'''
    def __init__(self, hmm, obs):
        self.hmm = hmm
        self.obs = obs 

        self.t = self.obs.size
        self.s = self.hmm.states_amount

        self.alpha = np.zeros((self.t, self.s))
        self.beta = np.ones((self.t, self.s))
        self.scales = []


    def __scale(self, arr) -> float:
        scale = 1 / arr.sum()
        self.scales.append(scale)

        arr *= scale


    def __forward(self):
        y = self.obs[0]
        self.alpha[0] = self.hmm.initial * self.hmm.out[:, y]
        self.__scale(self.alpha[0])

        for time in range(1, self.t):
            self.alpha[time] = np.dot(self.alpha[time-1], self.hmm.trans)

            y = self.obs[time]
            self.alpha[time] *= self.hmm.out[:, y]
            self.__scale(self.alpha[time])


    def __backward(self):
        self.beta[-1] *= self.scales[-1]

        for time in range(self.t-2, -1, -1):
            for u1 in range(self.s):
                y = self.obs[time+1]
                self.beta[time, u1] = np.array([self.hmm.trans[u1, u2] * self.hmm.out[u2, y] * self.beta[time+1, u2] 
                                     for u2 in range(self.s)]).sum()

            self.beta[time] *= self.scales[time]
    

    def fba(self):
        self.__forward()
        self.__backward()