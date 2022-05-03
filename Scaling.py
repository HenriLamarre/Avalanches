import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import time

sys.path.insert(0, '../')
import av

# start = time.time()
# avalanche1 = av.av_model(101, Nx = 48, Ny = 48)
# avalanche1.sigma2 = -1
# avalanche1.eps_drive = 1e-5
# avalanche1.name = 'D'
# avalanche1 = av.read_state('D48')
# avalanche1.do_soc(Niter=int(1e6), doplot=1, finish_with_soc=False, verbose=False)
# np.savez('/home/hlamarre/PycharmProjects/AVFS/Code/Python/TestingFilesHenri/Analysis/F148SOC.npz',
#           er=avalanche1.rel_e, el=avalanche1.lat_e)


# av.save_state(avalanche1, 'D48')


start = time.time()
# avalanche2 = av.av_model(101, Nx = 48, Ny = 48)
# avalanche2.sigma2 = -1
# avalanche2.eps_drive = 1e-5
# avalanche2.name = 'F1'
avalanche2 = av.read_state('F1_48_SOC')
avalanche2.do_soc(Niter=int(1e5), doplot=1, finish_with_soc=False, verbose=False)
# np.savez('/home/hlamarre/PycharmProjects/AVFS/Code/Python/TestingFilesHenri/Analysis/F148SOC.npz',
#           er=avalanche2.rel_e, el=avalanche2.lat_e)


av.save_state(avalanche2, 'F1_48_SOC')
print(time.time()-start)