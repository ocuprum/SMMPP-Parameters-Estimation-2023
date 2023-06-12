import numpy as np
from learning.Learner import Learn
from decoding.Decoder import Decoder
from show.Show import Show


REP_FIG = 'figures'
REP_TXT = 'txt'
LFILENAME = 'learning_result.txt'
DFILENAME = 'decoding_result.txt'


# Задаємо необхідні параметри для навчання 
INITIAL_DIST = np.array([0.3, 0.7]) # початковий розподіл

emb_trans = np.zeros((2, 2))          # матриця перехідних ймовірностей 
emb_trans[0, :] = [0, 1]              # вкладеного ланцюга
emb_trans[1, :] = [1, 0]
cond_soj_dist = np.zeros((2, 2, 3))               # умовні розподіли часу 
cond_soj_dist[0, 1, :] = [0.2, 0.4, 0.4]          # проведеного у станах
cond_soj_dist[1, 0, :] = [0.1, 0.2, 0.7]
KERNEL = np.zeros((2, 2, 3))                                   # напімарковське ядро
KERNEL[0, 1, :] = emb_trans[0, 1] * cond_soj_dist[0, 1, :]
KERNEL[1, 0, :] = emb_trans[1, 0] * cond_soj_dist[1, 0, :]

LAMS = np.array([5, 2])               # інтенсивності двічі стохастичного процесу Пуассона

MAX_SOJ_TIME = 3                      # максимальний час, який ланцюг може перебувати у стані

OBS_AMOUNT = 10                  # кількість послідовностей спостережень
MAX_TIME = [100]                        # часовий інтервал, на якому отримуємо спостереження
EPOCHS = 200                          # кількість ітерацій алгоритму Баума-Велша


# Генеруємо спостереження, початкову модель та знаходимо оцінку параметрів моделі та ДСПП
L = Learn(initial_dist=INITIAL_DIST, kernel=KERNEL, real_lams=LAMS,
          max_soj_time=MAX_SOJ_TIME, obs_amount=OBS_AMOUNT, test_max_time=MAX_TIME,
          outfile=REP_TXT + '/' + LFILENAME)

l_result = L.learn(epochs=EPOCHS)

MAX_TIME = MAX_TIME[0]
if l_result['est_hmm'].lams[0] < l_result['est_hmm'].lams[1]:
    l_result['est_hmm'] = l_result['inv_hmm'].copy()


# Відновлюємо послідовність станів прихованого марковського ланцюга 
D = Decoder(initial_dist=INITIAL_DIST, kernel=KERNEL, real_lams=LAMS,
            max_time=MAX_TIME, hmm=l_result['est_hmm'].copy(), outfile=REP_TXT+'/decoding_result.txt')

d_result = D.decode()


# Демонструємо зміну оцінки та послідовність спостережень
s = Show(ests=l_result['ests'], obs=l_result['obs'][0][:50],
         ests_filename='ests.png', obs_filename='obs.png',
         rep=REP_FIG)