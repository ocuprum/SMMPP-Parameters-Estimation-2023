import numpy as np
from observation.SemiMarkovChain import SemiMarkovChain as SMC
from observation.Observation import Observation as Obs
from decoding.AltViterbi import AltViterbi


class Decoder:
    def __init__(self, initial_dist, kernel, real_lams, max_time,
                 hmm, outfile='decoding_result.txt'):
        
        self.smc = SMC(max_time=max_time, initial_dist=initial_dist, kernel=kernel)
        self.obs = Obs(max_time=max_time, smc=self.smc, lams=real_lams)
        self.obs.restart()
        self.obs_seq = self.obs.observations

        self.hmm = hmm

        self.__smc_to_mc()
        self.fhandle = open(outfile, 'w')


    def __smc_to_mc(self):
        jump_times = np.append(self.obs.jump_times, [self.smc.max_time])

        chain = []
        prev_jump_time = jump_times[0]
        for i in range(len(self.obs.lams_obs)):
            time = jump_times[i+1] - prev_jump_time
            lam = self.obs.lams_obs[i]

            for t in range(int(time)):
                state = lam * self.hmm.max_soj_time + t
                chain.append(state)
            prev_jump_time = jump_times[i+1]

        self.true_seq = np.array(chain).astype(int)

    
    def __write(self, title):
        interval = (48 - len(title)) // 2
        self.fhandle.write('-' * 50)
        self.fhandle.write('\n|' + ' ' * interval + title + ' ' * (48 - len(title) - interval) + '|\n')
        self.fhandle.write('-' * 50)


    def decode(self):
        av = AltViterbi(hmm=self.hmm, obs=self.obs.observations)
        decoded = av.decode()

        self.__write(title='REAL OBSERVATION')
        self.fhandle.write('\n{}\n\n'.format(self.true_seq))
        self.__write(title='DECODED')
        dec_perc = self.true_seq[self.true_seq == decoded].size / self.true_seq.size * 100
        self.fhandle.write('\n-----> AltViterbi decoded {}%\n'.format(dec_perc))
        self.fhandle.write('\n{}\n\n'.format(decoded))

        self.fhandle.close()

        result = {'real': self.true_seq, 'decoded': decoded, 'perc': dec_perc, 'seq_len': self.true_seq.size}
        return result
