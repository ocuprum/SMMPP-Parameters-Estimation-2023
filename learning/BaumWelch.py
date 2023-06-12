import numpy as np
from evaluation.FBA import FBA


class BaumWelch:
    def __init__(self, hmm, obs):
        self.hmm = hmm
        self.obs = obs
        self.G = len(self.obs)


    def __est_initial(self):
        self.hmm.initial.fill(0)

        numer, denom = 0, 0

        for g in range(self.G):
            numer += self.fbas[g].alpha[0, 0::self.hmm.max_soj_time] * self.fbas[g].beta[0, 0::self.hmm.max_soj_time]
            den = self.fbas[g].alpha[0, :] * self.fbas[g].beta[0, :] 
            denom += den.sum()
        self.hmm.initial[0::self.hmm.max_soj_time] = numer / denom


    def __est_trans(self):
        for ind1 in range(self.fbas[0].s):
                denom = 0
                for g in range(self.G):
                    for time in range(self.fbas[g].t-1):
                        y = self.obs[g][time+1]
                        denom += self.fbas[g].alpha[time, ind1] * np.array([self.A[ind1, u] * self.B[u, y] * self.fbas[g].beta[time+1, u]
                                        for u in range(self.fbas[g].s)]).sum()
                    
                # (i, d) -> (i, d+1)
                if ind1 % self.hmm.max_soj_time != self.hmm.max_soj_time - 1:
                    ind2 = ind1 + 1
                    
                    numer = 0
                    for g in range(self.G):
                        for time in range(self.fbas[g].t-1):
                            y = self.obs[g][time+1]
                            numer += self.fbas[g].alpha[time, ind1] * self.A[ind1, ind2] * self.B[ind2, y] * self.fbas[g].beta[time+1, ind2]
                    
                    self.hmm.trans[ind1, ind2] = numer / denom

                # (i, d) -> (j, 0) 
                for ind2 in range(0, self.fbas[0].s, self.hmm.max_soj_time):
                    if ind1 != ind2 and ind1 // self.hmm.max_soj_time != ind2 // self.hmm.max_soj_time:

                        numer = 0
                        for g in range(self.G):
                            for time in range(self.fbas[g].t-1):
                                y = self.obs[g][time+1] 
                                numer += self.fbas[g].alpha[time, ind1] * self.A[ind1, ind2] * self.B[ind2, y] * self.fbas[g].beta[time+1, ind2]
                        self.hmm.trans[ind1, ind2] = numer / denom


    def __est_out(self):
        self.hmm.lams = np.zeros(self.X)

        for j in range(self.X):
            numer, denom = 0, 0

            for g in range(self.G):
                for time in range(self.fbas[g].t-1):
                    y = self.obs[g][time+1]

                    for d in range(self.hmm.max_soj_time):
                        ind1 = j * self.hmm.max_soj_time + d
                        val = self.fbas[g].alpha[time, ind1] * np.array([self.A[ind1, ind2] * self.B[ind2, y] * self.fbas[g].beta[time+1, ind2]
                                            for ind2 in range(self.fbas[g].s)]).sum()
                    y = self.obs[g][time]
                    numer += y * val
                    denom += val

            self.hmm.lams[j] = numer / denom

        self.hmm.compute_out()


    def estimate(self, epochs, epsilon=0.00001, minepochs=20):
        ests = []
        prev_est = None

        for e in range(epochs):
            self.fbas = []

            for g in range(self.G):
                self.fbas.append(FBA(hmm=self.hmm, obs=self.obs[g]))
                self.fbas[g].fba()

            est = - sum([sum(np.log(self.fbas[g].scales)) for g in range(self.G)])
            ests.append(est)
            if e > minepochs and abs((prev_est - est) / prev_est) < epsilon: 
                return e, np.array(ests)

            self.X = self.fbas[0].s // self.hmm.max_soj_time

            # Копіюємо поточну оцінку параметрів моделі
            self.A = np.copy(self.hmm.trans)
            self.B = np.copy(self.hmm.out)

            # Переоцінюємо параметри моделі
            self.__est_initial()
            self.__est_trans()
            self.__est_out()

            prev_est = est
            
        return epochs, ests