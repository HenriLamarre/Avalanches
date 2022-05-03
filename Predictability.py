import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing
import math
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
import av


def process_(number):
    np.random.seed(number)
    avalanche1 = av.read_state('F_pred')
    avalanche1.do_soc(Niter=int(tau*20), doplot=0, finish_with_soc=False, verbose=False, i_idum=number)
    comp_er = [0]
    for item in avalanche1.rel_e:
        if item:
            comp_er[-1] += item
        else:
            comp_er.append(0)
    comp_er[-1] = 0
    return comp_er

if __name__ == '__main__':
    start = time.time()
    # tau = 562
    tau = 2080
    n_jobs = 1000
    window = 40
    results = []
    # ers = []
    # results = pqdm(range(n_jobs), process_, n_jobs)
    with multiprocessing.Pool() as pool:
        results = list(pool.map(process_, range(n_jobs)))
    length = int(math.ceil(max(map(len, results))/window)*window)

    for i in range(len(results)):
        len_i = len(results[i])
        for j in range(len_i, length):
            results[i].append(0)

    stats = [0,0]
    for item in results:
        c = np.where(np.array(item) > 1e4, 1, 0)
        if np.any(c[int(0.5*tau):int(0.6*tau)]):
            stats[0] += 1
            if np.any(c[:int(0.1*tau)]):
                stats[1] += 1
    print(stats, stats[1]/stats[0])

    stats_holder = np.zeros((len(results), int(length/window)))
    filtrs = np.zeros((3, int(length/window)))
    for i in range(len(results)):
        a = np.where(np.array(results[i]) > 1e3, 1, 0)
        b = np.where(np.array(results[i]) > 5e3, 1, 0)
        c = np.where(np.array(results[i]) > 1e4, 1, 0)
        for j in range(len(results[i])):
            filtrs[0][j//window] += a[j]
            filtrs[1][j//window] += b[j]
            filtrs[2][j//window] += c[j]
            stats_holder[i][j // window] += c[j]
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
    x_space = np.linspace(0, length/tau, int(length/window))
    ax.step(x_space, filtrs[0], color='red', alpha=0.9)
    ax.step(x_space, filtrs[1], color='purple', alpha=0.9)
    ax.step(x_space, filtrs[2], color='blue', alpha=0.9)
    # ax.axvspan(x_space[peak1-10], x_space[peak1+10], alpha=0.3, color='grey')
    # ax.axvspan(x_space[peak2 - 10], x_space[peak2 + 10], alpha=0.3, color='grey')
    # plt.text(x_space[peak1-10], max(filtrs[0]), 'A')
    # plt.text(x_space[peak2-10], max(filtrs[0]), 'B')
    # plt.title('{}% of Avalanches in B were found in A'.format(round(stats[1]/stats[0]*100)))
    plt.savefig('./figures/Predict_B.png')
    plt.show()