"""avalanche.py  Contains a basic Avalanche class used 
for D-dimensional magnetic lattice avalanche SOC models

Created 2021-09-07"""
__author__ = "Henri Lamarre"

import numpy as np
import Methods as met
import time


class avalanche():
    """D = dimensions of the lattice
    Z_c = Stability treshold
    N = number of elements in a lattice dimension (linear size)
    sigma1, sigma2 = bounds for the driving mechanism"""

    def __init__(self, D, N, sigma1, sigma2, images=False):
        self.D = D
        self.N = N
        self.sigma1 = sigma1
        self.sigma2 = sigma2

        # These quantities allow us to compute the right Z_c to find
        # the SOC state as well as the suggested time to get there
        db = (self.sigma1+self.sigma2)/2
        self.Z_c = db / 1e-4 * self.D * 6 / self.N**2
        self.sugg_time = int(self.N ** 2 * 1e4)

        # Lat_B is the main lattice information containing the magnetic field
        self.lat_B = np.zeros((self.N, self.N))
        # Lat_C is a placeholder lattice used in the steps to store information
        self.lat_C = np.zeros((self.N, self.N))
        # S is a useful constant
        self.s = self.D * 2 + 1
        # mat_history is used to store lat_B at every step
        self.mat_history = []
        # er is the energy released with an avalanche for every step
        self.er = []
        # el is the lattice energy at every step
        self.el = []
        # e0 is the lattice energy unit
        self.e0 = 2 * self.D / self.s * self.Z_c ** 2
        # If we should save images and at which rate
        self.images = images
        # keeps in mind which lattice element got updated
        # last for effeciency
        self.lattice_increment = None
        # Remember the smallest j and largest k of the avalanche for speed
        self.width_j = [1, self.N - 1]
        self.width_k = [1, self.N - 1]
        # keeps track if we are avalanching
        self.avalanching = False
        # Statistics
        self.a_P = []
        self.a_T = []
        self.a_E = []

    def a_statistic_init(self, step_time):
        """Initializes the statistics of an avalanche
        P is the peak energy
        E is the total energy
        T is the time duration"""
        self.avalanching = True

        self.a_P.append([])
        self.a_T.append([])
        self.a_E.append([])

        self.a_P[-1].append(0)
        self.a_E[-1].append(0)
        #beggining time of the avalanche
        self.a_T[-1].append(step_time)

    def save_state(self, path):
        '''Saves the state of lat_B to path. Useful if lat_B is in SOC'''
        np.savez(path, lat_B=self.lat_B)

    def step(self, step_time):
        """ Updates the lattice either by adding
        increments of magnetism or avalanching
        The time variable is used to make the movie.
        it specifies which frames should be saved."""
        e = 0
        # This if statement checks if the last step was
        # a magnetism increment. If yes, then we only
        # need to check for avalanche around that point.

        if self.lattice_increment:
            range_list_j = range(self.lattice_increment[0] - 1, self.lattice_increment[0] + 1)
            range_list_k = range(self.lattice_increment[1] - 1, self.lattice_increment[1] + 1)

        # Otherwise if last step was avalanche, only need to check
        # Inside the avalanche
        else:
            range_list_j = range(self.width_j[0], self.width_j[1])
            range_list_k = range(self.width_k[0], self.width_k[1])

        # Initialize nearest neighbours
        NN = np.zeros((self.N, self.N))
        # Grid of nearest neighbours
        NN[1:-1, 1:-1] = self.lat_B[0:-2, 1:-1] + self.lat_B[1:-1, 0:-2] + self.lat_B[2:, 1:-1] + self.lat_B[1:-1, 2:]
        # Grid of curvatures
        Zk = self.lat_B - NN / 2 / self.D
        # Grid of 1 if unstable and 0 if stable
        unst = np.where(Zk > self.Z_c, 1, 0)
        # unstable points decrease and neighbours increase
        self.lat_C += -unst*self.Z_c * 2 * self.D / self.s +\
                    np.roll(unst, 1, axis=0) * self.Z_c / self.s + \
                    np.roll(unst, 1, axis=1) * self.Z_c / self.s + \
                    np.roll(unst, -1, axis=0) * self.Z_c / self.s + \
                    np.roll(unst, -1, axis=1) * self.Z_c / self.s
        # Energy dissipated in the process
        e = np.sum(2 * self.D / self.s * (2 * np.abs(Zk[Zk > self.Z_c]) / self.Z_c - 1) * self.Z_c ** 2)

            # If there is an avalanche
        if e > 0:
            if not self.avalanching:
                self.a_statistic_init(step_time)

            # update peak energy release
            if self.a_P[-1] < e/self.e0:
                self.a_P[-1] = e/self.e0
            # Update total energy release
            self.a_E[-1] += e/self.e0

            # Reset the width of the avalanche
            (nzx, nzy) = np.nonzero(self.lat_C)
            self.width_j[0] = min(nzx[nzx> 0])
            self.width_j[1] = max(nzx[nzx< self.N])
            self.width_k[0] = min(nzy[nzy > 0])
            self.width_k[1] = max(nzy[nzy < self.N])

            # updates the lattice array
            self.lat_B[1:-1, 1:-1] += self.lat_C[1:-1, 1:-1]
            self.lat_C = np.zeros((self.N, self.N))

            # makes sure that we check every lattice element in next step
            self.lattice_increment = None
        # Otherwise, add increment to grid
        else:
            # End of the avalanche
            if self.avalanching:
                self.avalanching = False
                # Ending time of the avalanche
                self.a_T[-1][0] = step_time - self.a_T[-1][0]
            # Find random lattice element
            randx, randy = np.random.randint(1, high=self.N - 1), np.random.randint(1, high=self.N - 1)
            # Random magnetism increment
            dB = np.random.random() * np.abs(self.sigma1 - self.sigma2) + self.sigma1
            # Update lattice
            self.lat_B[randx][randy] += dB
            # Next step has only to check the curvature of this point
            self.lattice_increment = [randx, randy]

        # Save the released energy
        self.er.append(e / self.e0)
        # save the lattice energy
        self.el.append(np.sum(self.lat_B ** 2))
        # Save images if we want them
        if self.images:
            # Save images with frequency 1/self.images
            if not step_time % self.images:
                list_holder = list(np.copy(self.lat_B))
                self.mat_history.append(list_holder)

    def loop(self, total_time, frequency, save=False, load=False):
        """Run the avalanche simulation for total_Time iterations.
        print the loading bar at 1/frequency iterations. Prints the time
        it took to run at the end of the loop.
        if save, saves lat_B to 'save'
        if load, loads the numpy array 'load' as lat_B"""
        if load:
            self.lat_B = np.load(load)['lat_B']
        start = time.time()
        for i in range(int(total_time)):
            if not i % int(frequency):
                print(str(int(i / time_ * 100)) + '%')
            self.step(i)
        end = time.time()
        if save:
            self.save_state(save)
        print('loop took '+str(round(end - start, 2)) + 's')


if __name__ == '__main__':
    avalanche1 = avalanche(2, 32, -0.2, 0.8, images=False)
    time_ = int(avalanche1.sugg_time*3/2)
    avalanche1.loop(time_, 10000, save = '/home/hlamarre/PycharmProjects/Avalanches/Saves/N32_SOC.npz',
                    load = '/home/hlamarre/PycharmProjects/Avalanches/Saves/N32_SOC.npz')
    met.plot_energies(avalanche1.el, avalanche1.er, time_, avalanche1.e0)
    # We ignore the last avalanche because the stats are truncated
    avalanche1.a_E.pop()
    avalanche1.a_P.pop()
    avalanche1.a_T.pop()
    met.plot_statistics(avalanche1.a_E, avalanche1.a_P, avalanche1.a_T)
