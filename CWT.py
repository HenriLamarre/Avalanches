import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import time

sys.path.insert(0, '../')
import av

start = time.time()
# avalanche1 = av.av_model(101, Nx = 128, Ny = 128)
# avalanche1.name = 'F1'
# avalanche1.sigma2 = -1
# avalanche1.eps_drive = 1e-5
# avalanche1 = av.read_state('F1')
# avalanche1.do_soc(Niter=int(1e6), doplot=1, finish_with_soc=False, verbose=False)
# # print(avalanche1.B)
# np.savez('/home/hlamarre/PycharmProjects/Avalanches/interactive/Saves/N32_F_32.npz',
#           lat_B = avalanche1.B) 
# av.save_state(avalanche1, 'F1')
# np.savez('/home/hlamarre/PycharmProjects/Avalanches/interactive/Saves/AVSF_F3.npz',
#           er=avalanche1.rel_e, el=avalanche1.lat_e)
# print('took ' + str(round(time.time()-start,2)) + ' seconds')

start = time.time()
avalanche2 = av.av_model(101, Nx = 32, Ny = 32)
avalanche2.name = 'F0'
avalanche2.sigma2 = -1
avalanche2.eps_drive = 1e-5
avalanche2 = av.read_state('F0')
avalanche2.do_soc(Niter=int(1e6), doplot=1, finish_with_soc=False, verbose=False)
# print(avalanche2.B)
# np.savez('/home/hlamarre/PycharmProjects/Avalanches/interactive/Saves/N32_F_32.npz',
#           lat_B = avalanche2.B)
av.save_state(avalanche2, 'F0')
# np.savez('/home/hlamarre/PycharmProjects/Avalanches/interactive/Saves/AVSF_F3.npz',
#           er=avalanche1.rel_e, el=avalanche2.lat_e)
print('took ' + str(round(time.time()-start,2)) + ' seconds')