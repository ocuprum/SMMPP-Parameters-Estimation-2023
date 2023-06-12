import numpy as np


delta_min, delta_max = 2 ** (-1022), -(2 ** 1022)

class AltViterbi:

    def __init__(self, hmm, obs):
        self.obs = obs

        self.hmm = hmm
        self.hmm.obs_max = np.max(self.obs)
        self.hmm.compute_out()

        self.t = self.obs.size
        self.s = self.hmm.initial.size

        self.delta = np.zeros((self.t, self.s))
        self.psi = np.zeros((self.t, self.s))


    def __preprocess(self):
        prep = lambda el: np.log(el) if el > delta_min else delta_max
        prep = np.vectorize(prep)

        mu = np.copy(self.hmm.initial)
        A = np.copy(self.hmm.trans)

        self.hmm.initial = prep(mu)
        self.hmm.trans = np.apply_along_axis(prep, 1, A)


    def __initialization(self):
        self.B = np.copy(self.hmm.out)

        y = self.obs[0]

        self.hmm.out[:, y] = np.log(self.B[:, y])
        self.delta[0] = self.hmm.initial + self.hmm.out[:, y]
        self.psi[0] = np.zeros(self.s)


    def __alt_step(self, time):
        y = self.obs[time]

        self.hmm.out[:, y] = np.log(self.B[:, y])
        for state in range(self.s):
            temp = self.delta[time-1] + self.hmm.trans[:, state].T
            self.delta[time, state] = self.hmm.out[state, y] + np.max(temp) 
            self.psi[time, state] = np.argmax(temp)
    

    def decode(self):     
        self.__preprocess()
        self.__initialization()

        for time in range(1, self.t):
            self.__alt_step(time)

        self.psi = self.psi.astype('int')
        decoded = []
        decoded.append(int(np.argmax(self.delta[-1])))

        for time in range(self.t-1, 0, -1):
            x_prev = decoded[-1]
            decoded.append(self.psi[time, x_prev])
        
        return np.array(decoded[::-1]).astype('int')
