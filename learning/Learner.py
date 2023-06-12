import numpy as np
from observation.SemiMarkovChain import SemiMarkovChain as SMC
from observation.Observation import Observation as Obs
from model.MarkovChain import MarkovChain as MC
from model.HMM import HMM
from learning.BaumWelch import BaumWelch as BW


class Learn:
    def __init__(self, initial_dist, kernel, real_lams,
                 max_soj_time, obs_amount, test_max_time,
                 outfile='learning_result.txt'):
        
        self.fhandle = open(outfile, 'w')

        self.test_max_time = test_max_time
        max_time = max(self.test_max_time)

        self.smc = SMC(max_time=max_time, initial_dist=initial_dist, kernel=kernel)
        self.obs = Obs(max_time=max_time, smc=self.smc, lams=real_lams)
        self.__get_obs(obs=self.obs, obs_amount=obs_amount)
        
        self.mc = MC(smc=self.smc)

        self.__write(title='TRUE PARAMETERS')
        self.fhandle.write(self.mc.__repr__())
        self.fhandle.write('\nLAMS: \n{}\n\n'.format(real_lams))

        self.__get_start_hmm(max_soj_time=max_soj_time)

        
    def __write(self, title):
        interval = (48 - len(title)) // 2
        self.fhandle.write('-' * 50)
        self.fhandle.write('\n|' + ' ' * interval + title + ' ' * (48 - len(title) - interval) + '|\n')
        self.fhandle.write('-' * 50)


    def __get_obs(self, obs, obs_amount):
        self.OBSERVATIONS = []

        for _ in range(obs_amount):
            obs.restart()
            self.OBSERVATIONS.append(obs.observations)


    def __compute_start_lams(self):
        lam = 0
        for seq in self.OBSERVATIONS:
            lam += np.mean(seq)

        return lam / len(self.OBSERVATIONS)
    

    def __get_start_hmm(self, max_soj_time):
        lam = self.__compute_start_lams()
        obs_max = np.max(self.OBSERVATIONS)

        self.hmm = HMM(lams=np.array([lam, lam]), max_soj_time=max_soj_time, obs_max=obs_max)
        self.start_hmm = self.hmm.copy()
        
        self.__write(title='START MODEL PARAMETERS')
        self.fhandle.write(self.hmm.__repr__())
        

    def __inverse_model(self):
        hmm_copy = self.bw.hmm.copy()

        mu = np.copy(self.bw.hmm.initial)
        A = np.copy(self.bw.hmm.trans.round(5))
        B = np.copy(self.bw.hmm.out)

        hmm_copy.initial[0], hmm_copy.initial[3] = mu[3], mu[0]

        hmm_copy.trans[0][1], hmm_copy.trans[0][3] = A[3][4], A[3][0]
        hmm_copy.trans[1][2], hmm_copy.trans[1][3] = A[4][5], A[4][0]
        hmm_copy.trans[3][0], hmm_copy.trans[3][4] = A[0][3], A[0][1]
        hmm_copy.trans[4][0], hmm_copy.trans[4][5] = A[1][3], A[1][2]

        hmm_copy.out[:3] = B[3:]
        hmm_copy.out[3:] = B[:3]

        hmm_copy.lams = np.zeros(2)
        hmm_copy.lams[0], hmm_copy.lams[1] = self.bw.hmm.lams[1], self.bw.hmm.lams[0]
        return hmm_copy


    def learn(self, epochs) -> None:
        for amount in self.test_max_time:
            hmm_copy = self.hmm.copy()

            OBSERVATIONS = [o[:amount] for o in self.OBSERVATIONS]

            self.bw = BW(hmm=hmm_copy, obs=OBSERVATIONS)

            ep, ests = self.bw.estimate(epochs=epochs)

            self.__write(title='MODEL PARAMETERS ESTIMATION')
            data_line = '\n-----> MAX_TIME: {}, OBS_AMOUNT: {}, EPOCHS: {}'.format(amount, len(self.OBSERVATIONS), ep)
            self.fhandle.write(data_line)
            self.fhandle.write(self.bw.hmm.__repr__())

            self.__write(title='INVERSED MODEL PARAMETERS ESTIMATION')
            inversed_hmm = self.__inverse_model()
            self.fhandle.write(inversed_hmm.__repr__())

            self.__write(title='OBSERVATIONS')
            for num, seq in enumerate(OBSERVATIONS):
                self.fhandle.write('\nSEQUENCE {}\n'.format(num))
                self.fhandle.write('{}\n'.format(seq))

        self.fhandle.close()
        result = {'ep': ep, 'ests': ests, 
                  'start_hmm': self.start_hmm, 'est_hmm': self.bw.hmm, 
                  'inv_hmm': inversed_hmm, 'obs': OBSERVATIONS,
                  'seq_len': self.OBSERVATIONS[0].size, 'obs_amount': len(self.OBSERVATIONS)}
        return result