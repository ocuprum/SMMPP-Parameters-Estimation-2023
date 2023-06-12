import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


class Show:
    def __init__(self, ests, obs, ests_filename, obs_filename, rep) -> None:
        self.__show_obs(obs, obs_filename, rep)
        self.__show_ests(ests, ests_filename, rep)

    
    def __show_ests(self, data, filename, rep):
        data = data[1:]
        iters = np.arange(1, data.size+1, 1)

        self.__fig, self.__ax = plt.subplots()

        self.__ax.plot(iters, data, color='#BAA2FE')
        self.__ax.scatter(iters, data, color='#805ADC')

        plt.gca().spines[['right', 'top']].set_visible(False)

        plt.xlabel('Ітерація', fontsize=10, labelpad=7)
        plt.rcParams['text.usetex'] = True
        plt.ylabel('$\ln P_{\phi}(Y^{(0)} = y^{(0)}, \dots, Y^{(G)} = y^{(G)})$', fontsize=10)
        plt.grid(axis='y', color = '#03cfba', linestyle = '--', linewidth = 0.5)

        self.__save(rep, filename)

    
    def __show_obs(self, data, filename, rep):
        obs_num = np.arange(0, data.size, 1)

        self.__fig, self.__ax = plt.subplots()
        plt.bar(obs_num, data, color='#03cfba')

        self.__ax.set_yticks(np.arange(np.max(data) + 1))

        plt.gca().spines[['right', 'top']].set_visible(False)

        plt.xlabel('Номер спостереження у послідовності', fontsize=10, labelpad=7)
        plt.ylabel('Кількість моментів настання подій', fontsize=10, labelpad=7)
        plt.grid(axis='y', color = '#9b1f73', linestyle = '--', linewidth = 0.5)

        self.__save(rep, filename)


    def __save(self, rep, filename):
        self.__fig.savefig(rep + '/' + filename, bbox_inches='tight', pad_inches = 0.1)