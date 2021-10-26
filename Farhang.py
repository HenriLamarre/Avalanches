"""Farhang.py  Contains the Farhang energy optimization avalanche model

Created 2021-10-25"""
__author__ = "Henri Lamarre"

import numpy as np
import Methods as Met
import time
import scipy
import matplotlib.pyplot as plt


class Farhang:
    """ Avalanche model using the minimal energy principle """

    def __init__(self, d, n):
        self.D = d  # Dimension of the lattice
        self.N = n  # Number of lattice points
        self.Ac_avg = 1  # Average of the normal threshold distribution
        self.sigma = 0.01  # Width of the normal threshold distribution
        self.lat_B = np.zeros((self.N, self.N))  # Lat_B is the main lattice information containing the magnetic field
        self.lat_C = np.zeros((self.N, self.N))  # Lat_C is a placeholder lattice used in the steps to store information
        self.avalanching = False  # If there is an avalanche happening
        self.released_energies = []  # cumulative storage of released energies at every step
        self.lattice_energy = []  # same with lattice energy

    def current(self, lattice, i, j):
        """ Computes the electric current J at a lattice point i,j """
        if i == 0:  # These conditions check if the point we are looking is near an edge
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
        return 4*lattice[i, j] - a - b - c - d  # Computes the current using a derivative approximation

    def opt_x(self, x, Zc, r1, r2, r3, theta):
        """ Function that defines x, the optimization parameter. Is used in root finding. """
        a = r1 + r2 + r3
        return Zc * (r1**2 + r2**2 + r3**2 - a * x) - 5 / 32 * a * theta / (5 / 32 * theta + a * Zc) - x

    def e_total(self, lattice):
        """ Returns the total energy of a specified lattice """
        return np.sum(1/2*lattice[1:-1,1:-1]*(4*lattice[1:-1,1:-1] - lattice[1:-1,:-2]
                                              - lattice[2:,1:-1] - lattice[1:-1,2:] - lattice[:-2,1:-1]))

    def step(self):
        Zc = np.random.normal(1, 0.01)  # normal distribution of threshold
        e = 0  # default energy released
        for i in range(1, self.N-1):
            for j in range(1, self.N-1):  # Check curvature
                curv = self.lat_B[i, j] - 1/4 * (self.lat_B[i, j-1] + self.lat_B[i, j+1] +
                                                 self.lat_B[i-1, j] + self.lat_B[i+1, j])
                if curv > Zc:  # If curvature bigger than threshold
                    [r1, r2, r3] = np.random.uniform(0, 1, size=(3, 1))  # Stochastic redistribution
                    a = r1 + r2 + r3
                    # Theta is used in finding x that minimizes lattice energy
                    theta = r1 * (-2 * self.current(self.lat_B, i, j - 1) + 3 * self.lat_B[i, j - 1]) + \
                            r2 * (-2 * self.current(self.lat_B, i + 1, j) + 3 * self.lat_B[i + 1, j]) + \
                            r3 * (-2 * self.current(self.lat_B, i, j + 1) + 3 * self.lat_B[i, j + 1]) + \
                            a * (2 * self.current(self.lat_B, i - 1, j) - 3 * self.lat_B[i - 1, j])
                    x = scipy.optimize.root(self.opt_x, 1, args=(Zc, r1, r2, r3, theta))['x'][0]  # Finds the optimal x
                    # New lat is defined to check the condition that the energy difference is actually positive
                    new_lat = self.lat_B.copy()
                    new_lat[i, j] -= 4 / 5 * Zc
                    new_lat[i, j - 1] += 4 / 5 * r1 / (x + a) * Zc
                    new_lat[i, j + 1] += 4 / 5 * r3 / (x + a) * Zc
                    new_lat[i - 1, j] += 4 / 5 * x / (x + a) * Zc
                    new_lat[i + 1, j] += 4 / 5 * r2 / (x + a) * Zc
                    deltaE = self.e_total(new_lat) - self.e_total(self.lat_B)  # The energy difference
                    if deltaE < 0:  # If there should be an avalanche
                        self.lat_C[i, j] -= 4 / 5 * Zc  # Updates the avalanche lattice array
                        self.lat_C[i, j - 1] += 4 / 5 * r1 / (x + a) * Zc
                        self.lat_C[i, j + 1] += 4 / 5 * r3 / (x + a) * Zc
                        self.lat_C[i - 1, j] += 4 / 5 * x / (x + a) * Zc
                        self.lat_C[i + 1, j] += 4 / 5 * r2 / (x + a) * Zc
                        self.avalanching = True  # Will be avalanching

        if not self.avalanching:
            epsilon = np.random.uniform(1e-7, 1e-5)  # uniform random increment
            self.lat_B *= (1+epsilon)  # Increase deterministically the lattice

        else:
            e = -self.e_total(self.lat_C) + self.e_total(self.lat_B)  # Energy released by avalanche
            self.lat_B[1:-1, 1:-1] += self.lat_C[1:-1, 1:-1]  # Avalanches / Updates the lattice
            self.lat_C = np.zeros((self.N, self.N))  # Resets the avalanche array
            self.avalanching = False  # By default, stops avalanche
        # Updates the cumulative information
        self.released_energies.append(e)
        self.lattice_energy.append(self.e_total(self.lat_B))

    def loop(self, total_time, save=False, load=False):
        start = time.time()
        if load:  # Otherwise, if we have an initial array
            self.lat_B = np.load(load)['lat_B']
        else:
            self.lat_B[1:-1, 1:-1] += np.random.uniform(1e-1, 1, size=(self.N - 2, self.N - 2))  # Initial distribution
        for i in range(int(total_time)):  # The actual loop
            self.step()
        if save:  # Save to numpy array for loading in other runs
            np.savez(save, lat_B=self.lat_B)
        end = time.time()
        print('loop took '+str(round(end - start, 2)) + 's')


if __name__ == '__main__':
    avalanche1 = Farhang(2, 10)
    t_ = 1e6
    avalanche1.loop(t_)#, save = '/home/hlamarre/PycharmProjects/Avalanches/Saves/N52_Farhang.npz',
                    #load = '/home/hlamarre/PycharmProjects/Avalanches/Saves/N52_Farhang.npz')
    Met.plot_energies(avalanche1.lattice_energy, avalanche1.released_energies, t_, 1)
