import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import time

sys.path.insert(0, '../')
import av

# beg = time.time()
# print('Starting F1_32')
# avalanche1 = av.av_model(101, Nx = 32, Ny = 32)
# avalanche1.name = 'F1'
# avalanche1.sigma2 = -1
# avalanche1.eps_drive = 1e-5
# avalanche1.do_soc(Niter=int(1e5), doplot=1, finish_with_soc=False, verbose=False)
# print('Saving F1_32, took {} seconds'.format(round(time.time()-beg), 1))
# av.save_state(avalanche1, 'F1_32_SOC')

# beg = time.time()
# print('Starting F1_48', time.time())
# avalanche1 = av.av_model(101, Nx = 48, Ny = 48)
# avalanche1.name = 'F1'
# avalanche1.sigma2 = -1
# avalanche1.eps_drive = 1e-5
# avalanche1.do_soc(Niter=int(5e6), doplot=1, finish_with_soc=False, verbose=False)
# print('Saving F1_48, took {} seconds'.format(round(time.time()-beg), 1))
# av.save_state(avalanche1, 'F1_48_SOC')

beg = time.time()
print('Starting F1_96')
avalanche1 = av.av_model(101, Nx = 96, Ny = 96)
avalanche1.name = 'F1'
avalanche1.sigma2 = -1
avalanche1.eps_drive = 1e-5
avalanche1.do_soc(Niter=int(1e6), doplot=1, finish_with_soc=False, verbose=False)
av.save_state(avalanche1, 'F1_96_SOC')
print('Saving F1_96, took {} seconds'.format(round(time.time()-beg), 1))
for i in range(10):
    beg = time.time()
    print('Starting F1_96', i)
    avalanche1 = av.read_state('F1_96_SOC')
    avalanche1.do_soc(Niter=int(1e6), doplot=1, finish_with_soc=False, verbose=False)
    print('Saving F1_96, took {} seconds'.format(round(time.time()-beg), 1))
    av.save_state(avalanche1, 'F1_96_SOC')