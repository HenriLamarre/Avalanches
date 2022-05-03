import numpy as np
import matplotlib.pyplot as plt
import time
import os
import avalancher_interactive as ai
import Methods as met
import multiprocessing
import math
from pqdm.processes import pqdm


def process_(number):
    np.random.seed(number)
    avalanche1 = ai.Avalanche(2, 48)
    avalanche1.name = 'A2_shuf_True_physical_True'
    t_ = 1e4
    avalanche1.loop(t_, save=False, load='/home/hlamarre/PycharmProjects/Avalanches/interactive/Saves/',
                    shuffle=True, physical=True, append_lists=True)
    comp_er = [0]

    for item in avalanche1.energy_disp:
        if item:
            comp_er[-1] += item
        else:
            comp_er.append(0)
    comp_er[-1] = 0
    return comp_er

if __name__ == '__main__':
    start = time.time()
    tau = 562
    n_jobs = 10000
    results = []
    # ers = []
    # results = pqdm(range(n_jobs), process_, n_jobs)
    with multiprocessing.Pool() as pool:
        results = list(pool.map(process_, range(n_jobs)))
    length = int(math.ceil(max(map(len, results))/10)*10)

    for i in range(len(results)):
        len_i = len(results[i])
        for j in range(len_i, length):
            results[i].append(0)

    stats_holder = np.zeros((len(results), int(length/10)))
    filtrs = np.zeros((3, int(length/10)))
    for i in range(len(results)):
        a = np.where(np.array(results[i]) > 1e4, 1, 0)
        b = np.where(np.array(results[i]) > 5e4, 1, 0)
        c = np.where(np.array(results[i]) > 1e5, 1, 0)
        for j in range(len(results[i])):
            filtrs[0][j//10] += a[j]
            filtrs[1][j//10] += b[j]
            filtrs[2][j//10] += c[j]
            stats_holder[i][j // 10] += c[j]
    end = time.time()
    print('Job took ' + str(round(end - start, 2)) + 's')

    # peaks = filtrs[0].argpartition(-2)[-2:]
    # peak1 = min(peaks)
    # peak2 = max(peaks)
    # stats = [0,0]
    # for item in stats_holder:
    #     if np.any(item[peak2-10:peak2+10]):
    #         stats[0] += 1
    #         if np.any(item[:peak2-10]):
    #             stats[1]+=1

    fig, ax = plt.subplots()
    x_space = np.linspace(0, length/tau, int(length/10))
    ax.step(x_space, filtrs[0], color='red', alpha=0.9)
    ax.step(x_space, filtrs[1], color='purple', alpha=0.9)
    ax.step(x_space, filtrs[2], color='blue', alpha=0.9)
    # ax.axvspan(x_space[peak1-10], x_space[peak1+10], alpha=0.3, color='grey')
    # ax.axvspan(x_space[peak2 - 10], x_space[peak2 + 10], alpha=0.3, color='grey')
    # plt.text(x_space[peak1-10], max(filtrs[0]), 'A')
    # plt.text(x_space[peak2-10], max(filtrs[0]), 'B')
    # plt.title('{}% of Avalanches in B were found in A'.format(round(stats[1]/stats[0]*100)))
    plt.savefig('./figures/Farhang_A2_Predict.png')
    plt.show()