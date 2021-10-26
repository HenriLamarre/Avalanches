"""Farhang.py  Contains the Farhang energy optimization avalanche model

Created 2021-10-25"""
__author__ = "Henri Lamarre"

import numpy as np
import Methods as Met
import time
import scipy
import matplotlib.pyplot as plt


class Farhang:

    def __init__(self, d, n):
        self.D = d
        self.N = n
        self.Ac_avg = 1
        self.sigma = 0.01
        self.lat_B = np.zeros((self.N, self.N))  # Lat_B is the main lattice information containing the magnetic field
        self.lat_C = np.zeros((self.N, self.N))  # Lat_C is a placeholder lattice used in the steps to store information
        self.avalanching = False
        self.rangex = [1, self.N-1]
        self.rangey = [1, self.N-1]
        self.released_energies = []
        self.lattice_energy = []

    def current(self, lattice, i, j):
        if i == 0:
            a = 0
        else:
            a = lattice[i - 1, j]
        if i == self.N-1:
            b = 0
        else:
            b = lattice[i+1, j]
        if j == 0:
            c = 0
        else:
            c = lattice[i, j-1]
        if j == self.N-1:
            d = 0
        else:
            d = lattice[i, j+1]
        return 4*lattice[i, j] - a - b - c - d

    def opt_x(self, x, Zc, r1, r2, r3, theta):
        a = r1 + r2 + r3
        return Zc * (r1**2 + r2**2 + r3**2 - a * x) - 5 / 32 * a * theta / (5 / 32 * theta + a * Zc) - x

    def step(self, steptime):
        Zc = np.random.normal(1, 0.01)
        e = 0
        for i in range(self.rangex[0], self.rangex[1]):
            for j in range(self.rangey[0], self.rangey[1]):
                curv = self.lat_B[i, j] - 1/4 * (self.lat_B[i, j-1] + self.lat_B[i, j+1] +
                                                 self.lat_B[i-1, j] + self.lat_B[i+1, j])
                if curv > Zc:
                    [r1, r2, r3] = np.random.uniform(0, 1, size=(3, 1))
                    a = r1 + r2 + r3
                    old_energy = 1/2*self.lat_B[i, j]*self.current(self.lat_B, i, j)
                    self.lat_C[i, j] -= 4 / 5 * Zc
                    new_energy = self.lat_C[i, j]*self.current(self.lat_C, i, j)
                    if new_energy - old_energy < 0:
                        theta = r1 * (-2 * self.current(self.lat_B, i, j - 1) + 3 * self.lat_B[i, j - 1]) + \
                            r2 * (-2 * self.current(self.lat_B, i + 1, j) + 3 * self.lat_B[i + 1, j]) + \
                            r3 * (-2 * self.current(self.lat_B, i, j + 1) + 3 * self.lat_B[i, j + 1]) + \
                            a * (2 * self.current(self.lat_B, i - 1, j) - 3 * self.lat_B[i - 1, j])
                        x = scipy.optimize.root(self.opt_x, 1, args=(Zc, r1, r2, r3, theta))['x'][0]
                        # print('x: ' + str(x))
                        self.lat_C[i, j - 1] += 4 / 5 * r1 / (x + a) * Zc
                        self.lat_C[i, j + 1] += 4 / 5 * r3 / (x + a) * Zc
                        self.lat_C[i - 1, j] += 4 / 5 * x / (x + a) * Zc
                        self.lat_C[i + 1, j] += 4 / 5 * r2 / (x + a) * Zc
                        self.avalanching = True
                        e += old_energy - new_energy
                    else:# reset the lat_C we changed
                        self.lat_C[i, j] += 4 / 5 * Zc

        if not self.avalanching:
            epsilon = np.random.uniform(1e-7, 1e-5)
            self.lat_B *= (1+epsilon)

        else:
            self.lat_B[1:-1, 1:-1] += self.lat_C[1:-1, 1:-1]
            self.lat_C = np.zeros((self.N, self.N))
            self.avalanching = False
        self.released_energies.append(e)
        self.lattice_energy.append(np.sum(1/2*self.lat_B[1:-1,1:-1]*\
                                          (4*self.lat_B[1:-1,1:-1] - self.lat_B[1:-1,:-2] - self.lat_B[2:,1:-1] -
                                           self.lat_B[1:-1,2:] - self.lat_B[:-2,1:-1])))

    def loop(self, total_time):
        start = time.time()
        self.lat_B[1:-1, 1:-1] += np.random.uniform(1e-1, 1, size=(self.N-2, self.N-2))

        for i in range(int(total_time)):  # The actual loop
            self.step(i)
            # if i>1e5:
            #     if not np.amax(self.lat_B):
            #         break
            #     print('max lattice: ' + str(np.amax(self.lat_B)))
        end = time.time()
        print('loop took '+str(round(end - start, 2)) + 's')


if __name__ == '__main__':
    avalanche1 = Farhang(2, 10)
    t_ = 1e6
    avalanche1.loop(t_)
    # print(avalanche1.lat_B)
    Met.plot_energies(avalanche1.lattice_energy, avalanche1.released_energies, t_, 1)