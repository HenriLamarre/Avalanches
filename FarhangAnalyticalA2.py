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
        self.name = 'FarhangAnalyticalA2'
        self.avalanching = False
        self.Z_mean = 1
        self.Z_sigma = 0.01
        self.energy_disp = []
        self.energy_lat = []
        self.lattice_save = []
        self.last_time = time.time()

    def e_total(self, lattice):
        """ Returns the total energy of a specified lattice """
        return np.sum(np.multiply(lattice[1:-1, 1:-1], lattice[1:-1, 1:-1]))

    def step(self, step_time, saving_lattice=False):
        #Zc = np.random.normal(self.Z_mean, self.Z_sigma)
        Zc = self.Z_mean
        if saving_lattice:
            self.lattice_save.append(self.lat_B)
        self.lat_C = np.zeros((self.n, self.n))
        self.avalanching = False
        self.lat_B[:,0] = 0
        self.lat_B[:,-1] = 0
        self.lat_B[0] = 0
        self.lat_B[-1] = 0
        curv1 = np.zeros((self.n, self.n))  # curvature of the lattice initialization
        curv = np.zeros((self.n, self.n))  # curvature of the lattice initialization
        curv1[1:-1, 1:-1] = self.lat_B[1:-1, 1:-1] - 1 / 4 * (self.lat_B[1:-1, 0:-2] + self.lat_B[1:-1, 2:] +
                                                             self.lat_B[0:-2, 1:-1] + self.lat_B[2:, 1:-1])
        curv[1:-1, 1:-1] = curv1[1:-1, 1:-1] - 1 / 4 * (curv1[1:-1, 0:-2] + curv1[1:-1, 2:] +
                                                              curv1[0:-2, 1:-1] + curv1[2:, 1:-1])
        for i in range(self.n):
            for j in range(self.n):
                if i == 0 or i ==self.n-1 or j == 0 or j == self.n-1:
                    pass
                else:
                    if curv[i, j] > Zc:
                        rs = np.random.uniform(0, 1, size=(3, 1))[:,0]
                        a = np.sum(rs)
                        directions = np.array([[i - 1, j], [i, j + 1], [i + 1, j], [i, j - 1]])
                        np.random.shuffle(directions)
                        theta = rs[0]*self.lat_B[directions[2][0], directions[2][1]] +\
                                rs[1] * self.lat_B[directions[1][0], directions[1][1]] +\
                                rs[2] * self.lat_B[directions[0][0], directions[0][1]] - \
                                a * self.lat_B[directions[3][0], directions[3][1]]
                        x = (4*Zc/5*(rs[0]**2+rs[1]**2+rs[2]**2)-theta*a)/(theta+4*Zc/5*a)
                        C = 4*Zc/5
                        phi = rs[0]**2+rs[1]**2+rs[2]**2
                        double_der = -(4*C/(x+a)**3)*(theta - 3*C/2/(x+a*(phi-x*a)) - C*a/2)
                        energy_calc_lattice = copy.deepcopy(self.lat_B)
                        energy_calc_lattice[i, j] -= 4 / 5 * Zc
                        energy_calc_lattice[directions[2][0], directions[2][1]] += 4 / 5 * Zc * rs[0] / (x+a)
                        energy_calc_lattice[directions[1][0], directions[1][1]] += 4 / 5 * Zc * rs[1] / (x+a)
                        energy_calc_lattice[directions[0][0], directions[0][1]] += 4 / 5 * Zc * rs[2] / (x+a)
                        energy_calc_lattice[directions[3][0], directions[3][1]] += 4 / 5 * Zc * x / (x+a)
                        new_energy_dissipated = -self.e_total(energy_calc_lattice)+self.e_total(self.lat_B)

                        if new_energy_dissipated < 0:
                            #print('local energy negative' + str(double_der))
                            pass
                        if new_energy_dissipated >= 0:
                            #print('local energy positive ' + str(double_der))
                            self.lat_C[i, j] -= 4 / 5 * Zc
                            self.lat_C[directions[2][0], directions[2][1]] += 4 / 5 * Zc * rs[0] / (x + a)
                            self.lat_C[directions[1][0], directions[1][1]] += 4 / 5 * Zc * rs[1] / (x + a)
                            self.lat_C[directions[0][0], directions[0][1]] += 4 / 5 * Zc * rs[2] / (x + a)
                            self.lat_C[directions[3][0], directions[3][1]] += 4 / 5 * Zc * x / (x + a)
                            self.avalanching = True

        if self.avalanching:
            energy_dissipated = -self.e_total(self.lat_B + self.lat_C) + self.e_total(self.lat_B)
            if energy_dissipated > 0:
                self.lat_B += self.lat_C
            else:
                # epsilon = 10 ** np.random.uniform(-7, -5)
                epsilon = 1e-5
                self.lat_B *= (1 + epsilon)
                energy_dissipated = 0
                self.avalanching=False
                #print('energy negative')


        else:
            # epsilon = 10**np.random.uniform(-7, -5)
            epsilon = 1e-5
            self.lat_B *= (1+epsilon)
            energy_dissipated = 0

        self.energy_disp.append(energy_dissipated)
        self.energy_lat.append(np.sum(np.multiply(self.lat_B, self.lat_B)))

    def loop(self, total_time, save=False, load=False, saving_lattice=False, progress_bar=True):
        if progress_bar:
            start = time.time()
        if load:  # Otherwise, if we have an initial array
            self.lat_B = np.load(load + 'N{}_{}.npz'.format(self.n, self.name))['lat_B']
        else:
            self.lat_B[1:-1, 1:-1] += np.random.uniform(1e-1, 1, size=(self.n - 2, self.n - 2))  # Initial distribution
        for i in range(int(total_time)):  # The actual loop
            if not i % int(total_time/100) and progress_bar:  # Progress bar
                print(str(int(i / total_time * 100)) + '%' + '  Time since last iteration: ' + str(round(time.time() - self.last_time, 2)) + 's')
                self.last_time = time.time()
                if save:  # saves the array periodically
                    np.savez(save + 'N{}_{}.npz'.format(self.n, self.name), lat_B=self.lat_B)
            self.step(i, saving_lattice)
        if save:  # Save to numpy array for loading in other runs and statistics
            np.savez(save + 'N{}_{}.npz'.format(self.n, self.name), lat_B=self.lat_B)
            if saving_lattice:
                np.savez(save + 'N{}_{}_savedlat.npz'.format(self.n, self.name), lattice_save = self.lattice_save)
        if progress_bar:
            end = time.time()
            print('loop took '+str(round(end - start, 2)) + 's')


if __name__ == '__main__':
    avalanche1 = Avalanche(2, 32)
    t_ = 1e4
    avalanche1.loop(t_, save =  '/home/hlamarre/PycharmProjects/Avalanches/Saves/',
                    load = '/home/hlamarre/PycharmProjects/Avalanches/Saves/', saving_lattice=False)
    ax1 = plt.subplot(211)
    ax1.plot(avalanche1.energy_disp)
    ax1.set_ylabel('er')
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(avalanche1.energy_lat)
    ax2.set_ylabel('el')
    plt.savefig('energies.png')
    plt.show()
    #np.savez('N{}_{}_data.npz'.format(avalanche1.n, avalanche1.name), el=avalanche1.energy_lat,
             #er=avalanche1.energy_disp)
