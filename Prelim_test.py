import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import time

sys.path.insert(0, '../')
import av

avalanche1 = av.av_model(101, Nx = 32, Ny = 32)
avalanche1.name = 'F0'
avalanche1.sigma2 = -1
avalanche1.eps_drive = 1e-5
# avalanche1 = av.read_state('D')
avalanche1.do_soc(Niter=int(1e6), doplot=1, finish_with_soc=False, verbose=False)

# print(avalanche1.name)
# comp_er = [0]
# for item in avalanche1.rel_e:
#     if item:
#         comp_er[-1] += item
#     else:
#         comp_er.append(0)
# comp_er[-1] = 0
# a = np.where(np.array(comp_er) > 1e5, 1, 0)
# times = []
# holder = 0
# for i in range(len(a)):
#     if a[i]:
#         times.append(i-holder)
#         holder = i
# times.pop(0)
# print(times, np.mean(times), np.std(times))
# # av.save_state(avalanche1, 'F_pred')
