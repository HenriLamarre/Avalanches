import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import avalancher_interactive as av
plt.rcParams['figure.figsize'] = [10, 15]

job_list = []
names = ['A2']
for i in range(1):
    for j in range(2):
        for k in range(2):
            job_list.append([names[i], str(bool(j)), str(bool(k))])

# print(job_list)


def process_(index):
    np.random.seed(index)
    avalanche = av.Avalanche(2, 48)
    avalanche.interactive_step(1e6, load='/home/hlamarre/PycharmProjects/Avalanches/interactive/Saves/',
                               save='/home/hlamarre/PycharmProjects/Avalanches/interactive/Saves/',
                               curv_type=job_list[index][0], shuffle=job_list[index][1], append_lists=True,
                               physical = job_list[index][2])
    avalanche = None

with multiprocessing.Pool() as pool:
    pool.map(process_, range(len(job_list)))