"""Farhang.py  Contains the Farhang energy optimization avalanche model

Created 2021-10-25"""
__author__ = "Henri Lamarre"

import numpy as np
import Methods as Met
import time
import scipy
import copy
import matplotlib.pyplot as plt


class Avalanche:
    """ Avalanche model using the minimal energy principle """

    def __init__(self, d, n):
        self.d = d
        self.n = n
        self.lat_B = np.zeros((self.n, self.n))
        self.lat_C = np.zeros((self.n, self.n))
        self.name = 'Deterministic'
        self.avalanching = False
        self.Z_mean = 1
        self.Z_sigma = 0.01
        self.energy_disp = []
        self.energy_lat = []
        self.last_time = time.time()

    def step(self, step_time):
        Zc = np.random.normal(self.Z_mean, self.Z_sigma)
        self.lat_C = np.zeros((self.n, self.n))
        self.avalanching = False
        self.lat_B[:, 0] = 0
        self.lat_B[:, -1] = 0
        self.lat_B[0] = 0
        self.lat_B[-1] = 0
        curv = np.zeros((self.n, self.n))  # curvature of the lattice initialization
        curv[1:-1, 1:-1] = self.lat_B[1:-1, 1:-1] - 1 / 4 * (self.lat_B[1:-1, 0:-2] + self.lat_B[1:-1, 2:] +
                                                             self.lat_B[0:-2, 1:-1] + self.lat_B[2:, 1:-1])
        for i in range(self.n):
            for j in range(self.n):
                if i == 0 or i == self.n-1 or j == 0 or j == self.n-1:
                    pass
                else:
                    if curv[i, j] > Zc:
                        has_reached_positive_energy = True
                        energy_dissipated = 0
                        counter = 0
                        while energy_dissipated <= 0:
                            counter += 1
                            if counter > 5:
                                has_reached_positive_energy = False
                                break
                            rs = np.random.uniform(0, 1, (4,1))
                            a = np.sum(rs)
                            energy_calc_lattice = copy.deepcopy(self.lat_B)
                            energy_calc_lattice[i, j] -= 4 / 5 * Zc
                            energy_calc_lattice[i + 1, j] += 4 / 5 * Zc * rs[0] / a
                            energy_calc_lattice[i-1, j] += 4 / 5 * Zc * rs[1] / a
                            energy_calc_lattice[i, j+1] += 4 / 5 * Zc * rs[2] / a
                            energy_calc_lattice[i, j-1] += 4 / 5 * Zc * rs[3] / a
                            energy_dissipated = -np.sum(np.multiply(energy_calc_lattice, energy_calc_lattice)) +\
                                                np.sum(np.multiply(self.lat_B, self.lat_B))
                        if has_reached_positive_energy:
                            self.lat_C[i, j] -= 4/5*Zc
                            self.lat_C[i+1, j] += 4 / 5 * Zc * rs[0] / a
                            self.lat_C[i-1, j] += 4 / 5 * Zc * rs[1] / a
                            self.lat_C[i, j+1] += 4 / 5 * Zc * rs[2] / a
                            self.lat_C[i, j-1] += 4 / 5 * Zc * rs[3] / a
                            self.avalanching = True

        if self.avalanching:
            old_lattice = copy.deepcopy(self.lat_B)
            energy_dissipated = -np.sum(np.multiply(self.lat_B + self.lat_C, self.lat_B + self.lat_C)) +\
                                np.sum(np.multiply(old_lattice, old_lattice))
            if energy_dissipated > 0:
                self.lat_B += self.lat_C
            else:
                epsilon = 10 ** np.random.uniform(-7, -5)
                self.lat_B *= (1 + epsilon)
                energy_dissipated = 0
                print('energy negative')


        else:
            epsilon = 10**np.random.uniform(-7, -5)
            self.lat_B *= (1+epsilon)
            energy_dissipated = 0

        self.energy_disp.append(energy_dissipated)
        self.energy_lat.append(np.sum(np.multiply(self.lat_B, self.lat_B)))

    def loop(self, total_time, save=False, load=False):
        start = time.time()
        if load:  # Otherwise, if we have an initial array
            self.lat_B = np.load(load + 'N{}_{}.npz'.format(self.n, self.name))['lat_B']
        else:
            self.lat_B[1:-1, 1:-1] += np.random.uniform(1e-1, 1, size=(self.n - 2, self.n - 2))  # Initial distribution
        for i in range(int(total_time)):  # The actual loop
            if not i % int(total_time/100):  # Progress bar
                print(str(int(i / total_time * 100)) + '%' + '  Time since last iteration: ' + str(round(time.time() - self.last_time, 2)) + 's')
                self.last_time = time.time()
                if save:  # saves the array periodically
                    np.savez(save + 'N{}_{}.npz'.format(self.n, self.name), lat_B=self.lat_B)
            self.step(i)
        if save:  # Save to numpy array for loading in other runs and statistics
            np.savez(save + 'N{}_{}.npz'.format(self.n, self.name), lat_B=self.lat_B)
        end = time.time()
        print('loop took '+str(round(end - start, 2)) + 's')


if __name__ == '__main__':
    avalanche1 = Avalanche(2, 32)
    t_ = 1e6
    avalanche1.loop(t_, save =  '/home/hlamarre/PycharmProjects/Avalanches/Saves/',
                load = '/home/hlamarre/PycharmProjects/Avalanches/Saves/')
    ax1 = plt.subplot(211)
    ax1.plot(avalanche1.energy_disp)
    ax1.set_ylabel('er')
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(avalanche1.energy_lat)
    ax2.set_ylabel('el')
    plt.savefig('energies.png')
    plt.show()
