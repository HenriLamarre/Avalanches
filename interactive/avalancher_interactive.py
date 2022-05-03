"""Farhang.py  Contains the Farhang energy optimization avalanche model

Created 2021-10-25"""
__author__ = "Henri Lamarre"

import numpy as np
import time
import scipy
import copy
import matplotlib.pyplot as plt
import tools as tl
from scipy.optimize import curve_fit


class Avalanche:
    """ Avalanche model using the minimal energy principle """

    def __init__(self, d, n):
        self.d = d
        self.n = n
        self.lat_B = np.zeros((self.n, self.n))
        self.lat_C = np.zeros((self.n, self.n))
        self.name = 'FarhangAnalytical'
        self.avalanching = False
        self.Z_mean = 1
        self.Z_sigma = 0.01
        self.energy_disp = []
        self.energy_lat = []
        self.lattice_save = []
        self.last_time = time.time()
        self.success_list= []

    def current(self, lattice, pos):
        """ Computes the electric current J at a lattice point i,j """
        i = pos[0]
        j = pos[1]
        if i == 0:  # These conditions check if the point we are looking is near an edge
            a = 0
        else:
            a = lattice[i - 1, j]
        if i == self.n-1:
            b = 0
        else:
            b = lattice[i+1, j]
        if j == 0:
            c = 0
        else:
            c = lattice[i, j-1]
        if j == self.n-1:
            d = 0
        else:
            d = lattice[i, j+1]
        return 4*lattice[i, j] - a - b - c - d  # Computes the current using a derivative approximation

    def e_total(self, lattice):
        """ Returns the total energy of a specified lattice """
        return np.sum(1/2*np.multiply(lattice[1:-1, 1:-1],
                                      4*lattice[1:-1, 1:-1] - lattice[1:-1, :-2] -
                                      lattice[2:, 1:-1] - lattice[1:-1, 2:] - lattice[:-2, 1:-1]))

    def step(self, step_time, shuffle = True, curv_type='A2', append_lists=False, physical=True):
        #Zc = np.random.normal(self.Z_mean, self.Z_sigma)
        Zc = self.Z_mean
        self.lat_C = np.zeros((self.n, self.n))
        self.avalanching = False
        self.lat_B[:,0] = 0
        self.lat_B[:,-1] = 0
        self.lat_B[0] = 0
        self.lat_B[-1] = 0
        
        if curv_type == 'A2':
            curv1 = np.zeros((self.n, self.n))
            curv = np.zeros((self.n, self.n))
            curv1[1:-1, 1:-1] = self.lat_B[1:-1, 1:-1] - 1 / 4 * (self.lat_B[1:-1, 0:-2] + self.lat_B[1:-1, 2:] +
                                                                 self.lat_B[0:-2, 1:-1] + self.lat_B[2:, 1:-1])
            curv[1:-1, 1:-1] = curv1[1:-1, 1:-1] - 1 / 4 * (curv1[1:-1, 0:-2] + curv1[1:-1, 2:] +
                                                                 curv1[0:-2, 1:-1] + curv1[2:, 1:-1])
        elif curv_type == '9stencil':
            curv = np.zeros((self.n, self.n))
            curv[1:-1, 1:-1] = self.lat_B[1:-1, 1:-1] - 1 / 20 * (4*(self.lat_B[1:-1, 0:-2] + self.lat_B[1:-1, 2:] +
                                                                 self.lat_B[0:-2, 1:-1] + self.lat_B[2:, 1:-1]) + 
                                                                (self.lat_B[0:-2, 0:-2] + self.lat_B[0:-2, 2:] +
                                                                 self.lat_B[2:, 0:-2] + self.lat_B[2:, 2:]))
        elif curv_type == 'normal':
            curv = np.zeros((self.n, self.n)) 
            curv[1:-1, 1:-1] = self.lat_B[1:-1, 1:-1] - 1 / 4 * (self.lat_B[1:-1, 0:-2] + self.lat_B[1:-1, 2:] +
                                                                 self.lat_B[0:-2, 1:-1] + self.lat_B[2:, 1:-1])
        success = [0,0]
        for i in range(self.n):
            for j in range(self.n):
                if i == 0 or i ==self.n-1 or j == 0 or j == self.n-1:
                    pass
                else:
                    if curv[i, j] > Zc:
                        rs = np.random.uniform(0, 1, size=(3, 1))[:,0]
                        r1 = rs[0]
                        r2 = rs[1]
                        r3 = rs[2]
                        a = np.sum(rs)
                        directions = np.array([[i - 1, j], [i, j + 1], [i + 1, j], [i, j - 1]])
                        if shuffle:
                            np.random.shuffle(directions[0:4])
                        else:
                            smallest_neighbour = np.argmin([self.lat_B[i - 1, j], self.lat_B[i, j + 1],
                                                         self.lat_B[i + 1, j], self.lat_B[i, j - 1]])
                            directions = np.roll(directions, -smallest_neighbour, axis=0)
                            np.random.shuffle(directions[0:3])
                        theta = -r1*self.current(self.lat_B, directions[0]) -\
                                r2*self.current(self.lat_B, directions[1]) -\
                                r3*self.current(self.lat_B, directions[2]) + a*self.current(self.lat_B, directions[3])
                        x = (3.2*(r1**2+r2**2+r3**2)*Zc - a*theta)/(theta + 3.2*Zc*a)
                        if x<0 and physical:
                            x=0
                        energy_calc_lattice = copy.deepcopy(self.lat_B)
                        energy_calc_lattice[i, j] -= 4 / 5 * Zc
                        energy_calc_lattice[directions[2][0], directions[2][1]] += 4 / 5 * Zc * rs[0] / (x+a)
                        energy_calc_lattice[directions[1][0], directions[1][1]] += 4 / 5 * Zc * rs[1] / (x+a)
                        energy_calc_lattice[directions[0][0], directions[0][1]] += 4 / 5 * Zc * rs[2] / (x+a)
                        energy_calc_lattice[directions[3][0], directions[3][1]] += 4 / 5 * Zc * x / (x+a)
                        new_energy_dissipated = -self.e_total(energy_calc_lattice) + self.e_total(self.lat_B)

                        if new_energy_dissipated < 0:
                            success[1] += 1
                        if new_energy_dissipated >= 0:
                            success[0] += 1
                            self.lat_C[i, j] -= 4 / 5 * Zc
                            self.lat_C[directions[2][0], directions[2][1]] += 4 / 5 * Zc * rs[0] / (x + a)
                            self.lat_C[directions[1][0], directions[1][1]] += 4 / 5 * Zc * rs[1] / (x + a)
                            self.lat_C[directions[0][0], directions[0][1]] += 4 / 5 * Zc * rs[2] / (x + a)
                            self.lat_C[directions[3][0], directions[3][1]] += 4 / 5 * Zc * x / (x + a)
                            self.avalanching = True

        if self.avalanching:
            if append_lists:
                self.success_list.append(round(success[0]/np.sum(success),2))
            energy_dissipated = -self.e_total(self.lat_B + self.lat_C) + self.e_total(self.lat_B)
            if energy_dissipated > 0:
                self.lat_B += self.lat_C
                self.avalanching = False
            else:
                epsilon = 1e-5
                self.lat_B *= (1 + epsilon)
                energy_dissipated = 0
                self.avalanching=False
                # print('energy negative')
        else:
            epsilon = 1e-5
            self.lat_B *= (1+epsilon)
            energy_dissipated = 0

        if append_lists:
            self.energy_disp.append(energy_dissipated)
            self.energy_lat.append(self.e_total(self.lat_B))
            #self.lattice_save.append(self.lat_B)


    def loop(self, total_time, load=False, save=False, shuffle=True, curv_type='A2', append_lists=False, physical=True):
        if load:  # Otherwise, if we have an initial array
            self.lat_B = np.load(load + 'N{}_{}.npz'.format(self.n, self.name))['lat_B']
        else:
            self.lat_B[1:-1, 1:-1] += np.random.uniform(1e-1, 1, size=(self.n - 2, self.n - 2))  # Initial distribution
        for i in range(int(total_time)):  # The actual loop
            self.step(i, shuffle, curv_type, append_lists, physical)
            if save and not i%100:  # Save to numpy array for loading in other runs and statistics
                np.savez(save + 'N{}_{}.npz'.format(self.n, self.name), lat_B=self.lat_B)
        if save:  # Save to numpy array for loading in other runs and statistics
            np.savez(save + 'N{}_{}.npz'.format(self.n, self.name), lat_B=self.lat_B)

    def interactive_step(self, t_, save=False, load=False, shuffle=True,
                         curv_type='A2', append_lists=False, physical=True):
        self.name = curv_type + '_shuf_' + str(shuffle)+'_physical_'+str(physical)
        self.loop(t_, load, save, shuffle, curv_type, append_lists, physical)
        if append_lists:
            E,P,T = self.data_scanner()
            E_ = self.hist_(E)
            P_ = self.hist_(P)
            T_ = self.hist_(T)

            fig, ax = plt.subplots(4,2)
            curv = tl.curvature(self.lat_B, mode = curv_type, n=self.n)
            ax[0,1].matshow(self.lat_B)
            ax[0,1].set_title('lat_B')
            ax[1,1].matshow(curv)
            ax[1,1].set_title('CurvA2')
            ax[0,0].plot(self.energy_disp)
            ax[0,0].set_title('Energy Dissipated')
            ax[1,0].plot(self.energy_lat)
            ax[1,0].set_title('Lattice Energy')

            ax[2,0].scatter(np.log10(E_[1][E_[0][0] > 1e-10]), np.log10(E_[0][0][E_[0][0] > 1e-10]))
            #ax[2,0].plot(np.log10(E_[1]), self.line(np.log10(E_[1]), E_[2][0], E_[2][1]))
            ax[2,0].set_title('Energy')

            ax[2, 1].scatter(np.log10(P_[1][P_[0][0] > 1e-10]), np.log10(P_[0][0][P_[0][0] > 1e-10]))
            #ax[2, 1].plot(np.log10(P_[1]), self.line(np.log10(P_[1]), P_[2][0], P_[2][1]))
            ax[2, 1].set_title('Max Energy ')

            ax[3, 0].scatter(np.log10(T_[1][T_[0][0] > 1e-10]), np.log10(T_[0][0][T_[0][0] > 1e-10]))
            #ax[3, 0].plot(np.log10(T_[1]), self.line(np.log10(T_[1]), T_[2][0], T_[2][1]))
            ax[3, 0].set_title('Duration of avalanches ')

            plt.tight_layout()
            plt.savefig('./{}_shuf_{}_physical_{}.png'.format(self.name, shuffle, physical))
            #plt.show()
            print('Achieved {} %'.format(round(np.mean(self.success_list)*100, 2)))
            np.savez(save + 'N{}_{}_data.npz'.format(self.n, self.name), el = self.energy_lat, er = self.energy_disp)


    def data_scanner(self):
        E = []
        P = []
        T = []
        avalanching = False
        time_holder = 0
        energy_holder = 0
        peak_holder = 0
        for i in range(len(self.energy_disp)):
            if self.energy_disp[i] and not avalanching:
                avalanching = True
                time_holder = i
                energy_holder += self.energy_disp[i]
                peak_holder = self.energy_disp[i]
            if self.energy_disp[i] and avalanching:
                energy_holder += self.energy_disp[i]
                if self.energy_disp[i] > peak_holder:
                    peak_holder = self.energy_disp[i]
            if not self.energy_disp[i] and avalanching:
                avalanching = False
                T.append(i - time_holder)
                P.append(peak_holder)
                peak_holder = 0
                E.append(energy_holder)
                energy_holder = 0
        return E, P, T

    def line(self, x, a, b):
        return -a * x + b

    def hist_(self, array):
        space = np.logspace(np.log10(min(array)), np.log10(max(array)), 100, base=10)
        hist = np.histogram(array, bins=space[10:-15], density=True)
        for i in range(len(hist[0])):  # transforms 0 into small values for no errors
            if hist[0][i] < 1e-15:
                hist[0][i] = 1e-15
        x = (hist[1][:-1] + hist[1][1:]) / 2  # Average the bins sides for the fitting
        # refined_x = x[hist[0] > 1e-10]
        # refined_hist = hist[0][hist[0] > 1e-10]
        # popt, pcov = curve_fit(self.line, np.log10(refined_x[np.log10(refined_x)>0.5]),
        #                        np.log10(refined_hist[np.log10(refined_x)>0.5]))  # fitting of the line
        popt = []
        return [hist, x, popt]

if __name__ == '__main__':
    avalanche1 = Avalanche(2,32)
    t_ = 1e6
    avalanche1.name = 'F_32'
    avalanche1.loop(t_, save='/home/hlamarre/PycharmProjects/Avalanches/interactive/Saves',
                    load='/home/hlamarre/PycharmProjects/Avalanches/interactive/Saves/', shuffle=True, curv_type='normal',
                    append_lists=True, physical=False)
    # np.savez('saved_lattice.npz', lat = avalanche1.lattice_save)
    ax1 = plt.subplot(211)
    ax1.plot(avalanche1.energy_disp)
    ax1.set_ylabel('er')
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(avalanche1.energy_lat)
    ax2.set_ylabel('el')
    plt.savefig('energies.png')
    plt.show()