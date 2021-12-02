"""Farhang.py  Contains the Farhang energy optimization avalanche model

Created 2021-10-25"""
__author__ = "Henri Lamarre"

import numpy as np
import Methods as Met
import time
import scipy
import copy
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
        self.a_P = []  # Stores peak energy of avalanches
        self.a_T = []  # Stores time duration of avalanches
        self.a_E = []  # Stores total energy dissipated in avalanche
        self.last_time = 0
        self.curvs = []

    def a_statistic_init(self, step_time):
        """Initializes the statistics of an avalanche
        P is the peak energy
        E is the total energy
        T is the time duration"""

        self.a_P.append(0)  # Initialize peak energy of avalanche
        self.a_E.append(0)  # Initialize total energy of avalanche
        self.a_T.append(step_time)  # beginning time of the avalanche

    def current(self, lattice, pos):
        """ Computes the electric current J at a lattice point i,j """
        i = pos[0]
        j = pos[1]
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
        return 6.4*Zc * (r1**2+r2**2+r3**2-a*x)/(x+a) - theta


    def e_total(self, lattice):
        """ Returns the total energy of a specified lattice """
        return np.sum(1/2*np.multiply(lattice[1:-1, 1:-1],
                                      4*lattice[1:-1, 1:-1] - lattice[1:-1, :-2] -
                                      lattice[2:, 1:-1] - lattice[1:-1, 2:] - lattice[:-2, 1:-1]))

    def step(self, step_time):

        Zc = np.random.normal(1, 0.01)  # normal distribution of threshold
        e = 0  # default energy released
        curv = np.zeros((self.N, self.N))  # curvature of the lattice initialization
        curv[1:-1, 1:-1] = self.lat_B[1:-1, 1:-1] - 1 / 4 * (self.lat_B[1:-1, 0:-2] + self.lat_B[1:-1, 2:] +
                            self.lat_B[0:-2, 1:-1] + self.lat_B[2:, 1:-1])  # curvature computation

        if step_time > 0:  # saves the curve matrix for plotting.
            self.curvs.append(curv)

        prev_avalanching = bool(self.avalanching)  # remembers if the last iteration was avalanche or not
        self.avalanching = False  # defaults to false
        for i in range(1, self.N - 1):  # loops over lattice to check if avalanche should occur
            for j in range(1, self.N - 1):
                if curv[i][j] > Zc:  # condition to investigate optimization
                    [r1, r2, r3] = np.random.uniform(0, 1, size=(3, 1))  # Stochastic redistribution
                    a = r1 + r2 + r3
                    seq = np.array([[i,j+1], [i, j-1], [i-1, j], [i+1, j]])
                    np.random.shuffle(seq) # random direction for isotropy

                    # Theta is used in finding x that minimizes lattice energy
                    theta = r1* (-2 * self.current(self.lat_B, seq[0]) - self.lat_B[i, j]) +\
                            r2* (-2 * self.current(self.lat_B, seq[1]) - self.lat_B[i, j]) + \
                            r3 * (-2 * self.current(self.lat_B, seq[2]) - self.lat_B[i, j]) -\
                            a * (-2 * self.current(self.lat_B, seq[3]) - self.lat_B[i, j])
                    result = scipy.optimize.root(self.opt_x, 1, args=(Zc, r1, r2, r3, theta))
                    x = result['x'][0]  # Finds the optimal x
                    # New lat is defined to check the condition that the energy difference is actually positive
                    new_lat = copy.deepcopy(self.lat_B)
                    new_lat[i, j] -= 4 / 5 * Zc
                    new_lat[seq[0][0], seq[0][1]] += 4 / 5 * r1 / (x + a) * Zc
                    new_lat[seq[2][0], seq[2][1]] += 4 / 5 * r3 / (x + a) * Zc
                    new_lat[seq[3][0], seq[3][1]] += 4 / 5 * x / (x + a) * Zc
                    new_lat[seq[1][0], seq[1][1]] += 4 / 5 * r2 / (x + a) * Zc
                    deltaE = self.e_total(new_lat) - self.e_total(self.lat_B)  # The energy difference
                    if deltaE < 0:  # If there should be an avalanche
                        self.lat_C[i, j] -= 4 / 5 * Zc  # Updates the avalanche lattice array
                        self.lat_C[seq[0][0], seq[0][1]] += 4 / 5 * r1 / (x + a) * Zc
                        self.lat_C[seq[2][0], seq[2][1]] += 4 / 5 * r3 / (x + a) * Zc
                        self.lat_C[seq[3][0], seq[3][1]] += 4 / 5 * x / (x + a) * Zc
                        self.lat_C[seq[1][0], seq[1][1]] += 4 / 5 * r2 / (x + a) * Zc
                        self.avalanching = True



        if not self.avalanching:
            if prev_avalanching:  # stops the avalanche time duration
                self.a_T[-1] = step_time - self.a_T[-1]  # Ending time of the avalanche
            epsilon = np.random.uniform(1e-7, 1e-5)  # uniform random increment
            self.lat_B *= (1+epsilon)  # Increase deterministically the lattice

        else:
            if not prev_avalanching:  # new avalanche, we have to initialize the energy dissipation and peak energy
                self.a_statistic_init(step_time)
            old_lattice = copy.deepcopy(self.lat_B)  # old lattice for energy computation
            self.lat_B[1:-1, 1:-1] += self.lat_C[1:-1, 1:-1]  # Avalanches / Updates the lattice
            e = self.e_total(old_lattice) - self.e_total(self.lat_B)  # Energy released by avalanche step
            # if e<0:
            #     plt.figure()
            #     plt.imshow(self.lat_C)
            #     plt.show()
            if self.a_P[-1] < e:  # update peak energy release
                self.a_P[-1] = e
            self.a_E[-1] += e  # Update total energy release
            self.lat_C = np.zeros((self.N, self.N))  # Resets the avalanche array

        # Updates the cumulative information
        self.released_energies.append(e)
        self.lattice_energy.append(self.e_total(self.lat_B))

    def loop(self, total_time, save=False, load=False):
        start = time.time()
        if load:  # Otherwise, if we have an initial array
            self.lat_B = np.load(load + 'N{}_Farhang.npz'.format(self.N))['lat_B']
        else:
            self.lat_B[1:-1, 1:-1] += np.random.uniform(1e-1, 1, size=(self.N - 2, self.N - 2))  # Initial distribution
        for i in range(int(total_time)):  # The actual loop
            if not i % int(total_time/100):  # Progress bar
                print(str(int(i / total_time * 100)) + '%' + '  Time since last iteration: ' + str(round(time.time() - self.last_time, 2)) + 's')
                self.last_time = time.time()
                if save:  # saves the array periodically
                    np.savez(save + 'N{}_Farhang.npz'.format(self.N), lat_B=self.lat_B)
            self.step(i)
        if save:  # Save to numpy array for loading in other runs and statistics
            np.savez(save + 'N{}_Farhang.npz'.format(self.N), lat_B=self.lat_B)
            np.savez(save + 'N{}_Farhang_stats.npz'.format(self.N), e_l=avalanche1.lattice_energy,
                     e_r=avalanche1.released_energies, a_r=self.a_P, a_e=self.a_E, a_t=self.a_T)
            # np.savez(save + 'N{}_Farhang_curvs.npz'.format(self.N), curvs=self.curvs)
        end = time.time()
        print('loop took '+str(round(end - start, 2)) + 's')


if __name__ == '__main__':
    avalanche1 = Farhang(2, 32)
    t_ = 1e3
    avalanche1.loop(t_, save =  '/home/hlamarre/PycharmProjects/Avalanches/Saves/',
                    load = '/home/hlamarre/PycharmProjects/Avalanches/Saves/')
    Met.plot_energies(avalanche1.lattice_energy, avalanche1.released_energies, t_, 1)
