import numpy as np
import matplotlib.pyplot as plt
import time
import os
import FarhangAnalytical as FA
import Methods as met
from multiprocessing import Pool


def scanner(er):
    E = []
    times = []
    avalanching = False
    for i in range(len(er)):
        if er[i] and not avalanching:
            avalanching = True
            times.append(i)
            E.append(0)
            E[-1] += er[i]
        if er[i] and avalanching:
            E[-1] += er[i]
        if not er[i] and avalanching:
            avalanching = False
    return np.array(E), np.array(times)

def process_(save_):
    occurence = np.zeros((3, 30000))
    for i in range(100):
        print(i)
        avalanche1 = FA.Avalanche(2, 32)
        avalanche1.name = 'FarhangAnalytical3_predictability'
        t_ = 30000
        avalanche1.loop(t_, save=False, load='/home/hlamarre/PycharmProjects/Avalanches/Saves/', saving_lattice=False,
                        progress_bar=False)
        E, times = scanner(avalanche1.energy_disp)
        max_e = np.max(E)
        if max_e > 1e4:
            E_occ = times[E > 1e4]
            occurence[0][E_occ] += 1
        if max_e > 5e4:
            E_occ = times[E > 5e4]
            occurence[1][E_occ] += 1
        if max_e > 1e5:
            E_occ = times[E > 1e5]
            occurence[2][E_occ] += 1
    return occurence.tolist()

if __name__ == '__main__':
    start = time.time()
    pool = Pool(5)
    outputs = pool.map(process_, 'occurence')
    occurence = np.sum(outputs, axis=0)
    np.savez('occurence3.npz', occurence=occurence)
    end = time.time()
    print('Job took ' + str(round(end - start, 2)) + 's')


