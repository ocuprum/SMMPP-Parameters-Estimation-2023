import numpy as np
from observation.DSPP import DSPP

rng = np.random.default_rng()

class Observation:
    def __init__(self, max_time, smc, lams):
        # Ініціалізація напівмарковського ланцюга
        self.smc = smc
        self.smc.max_time = max_time
        
        self.jump_times = np.array([])
        
        self.lams=lams

        # Ініціалізація двічі стохастичного процесу Пуассона
        self.dspp = DSPP(lams=lams)
            
        # Ініціалізація масиву спостережень, масиву змін станів
        # та масиву часів настання подій
        self.observations = np.array([])
        self.lams_obs = np.array([])
        self.moments = np.array([])

        # Отримання спостережень
        self.__gen_obs()
    

    def __get_moments(self, prev_S):
        '''Метод, який генерує моменти настання події'''

        new_moments = rng.uniform(low=prev_S, high=self.smc.S, size=self.dspp.events_amount)
        self.moments = np.append(self.moments, new_moments)
        

    def __gen_obs(self):
        '''Метод, який генерує масив спостережень'''

        while self.smc.S < self.smc.max_time:
            prev_X = self.smc.X
            prev_S = self.smc.S
            self.jump_times = np.append(self.jump_times, prev_S)

            self.smc.next()

            self.dspp.next(smc_state=prev_X, prev_S=prev_S, cur_S=self.smc.S)
            self.lams_obs = np.append(self.lams_obs, prev_X)
            self.__get_moments(prev_S=prev_S)

        self.observations = np.array(
            [self.moments[(i <= self.moments) & (self.moments < i+1)].size 
             for i in range(self.smc.max_time)]
        )

    
    def restart(self):
        self.smc._restart()
        first_X = self.smc.X
        
        self.jump_times = np.array([])

        self.dspp = DSPP(lams=self.lams)
            
        self.observations = np.array([])
        self.lams_obs = np.array([])
        self.moments = np.array([])

        self.__gen_obs()
        return first_X