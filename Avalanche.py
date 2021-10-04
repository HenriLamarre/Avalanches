"""avalanche.py  Contains a basic Avalanche class used 
for D-dimensional magnetic lattice avalanche SOC models

Created 2021-09-07"""
__author__ = "Henri Lamarre"

import numpy as np
import Methods as Met
import time


class Avalanche:
    """d = dimensions of the lattice
    Z_c = Stability threshold
    n = number of elements in a lattice dimension (linear size)
    sigma1, sigma2 = bounds for the driving mechanism"""

    def __init__(self, d, n, sigma1, sigma2, images=False, nn_type='Normal'):
        self.D = d
        self.N = n
        self.sigma1 = sigma1
        self.sigma2 = sigma2

        # These quantities allow us to compute the right Z_c to find
        # the SOC state as well as the suggested time to get there
        db = (self.sigma1+self.sigma2)/2
        self.Z_c = db / 1e-4 * self.D * 6 / self.N**2
        self.sugg_time = int(self.N ** 2 * 1e4)
        self.nn_type = nn_type  # What type of nearest neighbours we use
        self.lat_B = np.zeros((self.N, self.N))  # Lat_B is the main lattice information containing the magnetic field
        self.lat_C = np.zeros((self.N, self.N))  # Lat_C is a placeholder lattice used in the steps to store information
        self.s = self.D * 2 + 1  # S is a useful constant
        self.mat_history = []  # mat_history is used to store lat_B at every step
        self.er = []  # er is the energy released with an avalanche for every step
        self.el = []  # el is the lattice energy at every step
        self.e0 = 2 * self.D / self.s * self.Z_c ** 2  # e0 is the lattice energy unit
        self.images = images  # If we should save images and at which rate
        self.avalanching = False  # keeps track if we are avalanching
        self.a_P = []  # Stores peak energy of avalanches
        self.a_T = []  # Stores time duration of avalanches
        self.a_E = []  # Stores total energy dissipated in avalanche

    def a_statistic_init(self, step_time):
        """Initializes the statistics of an avalanche
        P is the peak energy
        E is the total energy
        T is the time duration"""
        self.avalanching = True

        self.a_P.append([])  # New avalanche
        self.a_T.append([])
        self.a_E.append([])

        self.a_P[-1].append(0)  # Initialize peak energy of avalanche
        self.a_E[-1].append(0)  # Initialize total energy of avalanche
        self.a_T[-1].append(step_time)  # beginning time of the avalanche

    def save_state(self, path):
        """Saves the state of lat_B to path. Useful if lat_B is in SOC"""
        np.savez(path, lat_B=self.lat_B)

    def save_stats(self, path):
        """Saves the avalanche statistics to path"""
        np.savez(path, a_E=self.a_E, a_T=self.a_T, a_P=self.a_P)

    def step(self, step_time):
        """ Updates the lattice either by adding
        increments of magnetism or avalanching
        The time variable is used to make the movie.
        it specifies which frames should be saved."""

        nn = np.zeros((self.N, self.N))  # Initialize nearest neighbours
        # Grid of nearest neighbours
        nn[1:-1, 1:-1] = self.lat_B[0:-2, 1:-1] + self.lat_B[1:-1, 0:-2] + self.lat_B[2:, 1:-1] + self.lat_B[1:-1, 2:]
        zk = self.lat_B - nn / 2 / self.D  # Grid of curvatures
        unst = np.where(zk > self.Z_c, 1, 0)  # Grid of 1 if unstable and 0 if stable
        # unstable points decrease and neighbours increase
        self.lat_C += -unst*self.Z_c * 2 * self.D / self.s +\
            np.roll(unst, 1, axis=0) * self.Z_c / self.s + \
            np.roll(unst, 1, axis=1) * self.Z_c / self.s + \
            np.roll(unst, -1, axis=0) * self.Z_c / self.s + \
            np.roll(unst, -1, axis=1) * self.Z_c / self.s
        # Energy dissipated in the process
        e = np.sum(2 * self.D / self.s * (2 * np.abs(zk[zk > self.Z_c]) / self.Z_c - 1) * self.Z_c ** 2)

        # If there is an avalanche
        if e > 0:
            if not self.avalanching:
                self.a_statistic_init(step_time)  # initialize the statistics
            if self.a_P[-1] < e/self.e0:  # update peak energy release
                self.a_P[-1] = e/self.e0
            self.a_E[-1] += e/self.e0  # Update total energy release
            # updates the lattice array
            self.lat_B[1:-1, 1:-1] += self.lat_C[1:-1, 1:-1]
            self.lat_C = np.zeros((self.N, self.N))

        # If there is no avalanche
        else:  # Otherwise, add increment to grid
            if self.avalanching:  # End of the avalanche
                self.avalanching = False
                self.a_T[-1][0] = step_time - self.a_T[-1][0]  # Ending time of the avalanche
            # Find random lattice element
            randx, randy = np.random.randint(1, high=self.N - 1), np.random.randint(1, high=self.N - 1)
            db = np.random.random() * np.abs(self.sigma1 - self.sigma2) + self.sigma1  # Random magnetism increment
            self.lat_B[randx][randy] += db  # Update lattice

        self.er.append(e / self.e0)  # Save the released energy
        self.el.append(np.sum(self.lat_B ** 2))  # save the lattice energy
        if self.images:  # Save images if we want them
            if not step_time % self.images:  # Save images with frequency 1/self.images
                list_holder = list(np.copy(self.lat_B))
                self.mat_history.append(list_holder)

    def step_hex(self, step_time):
        """ Updates the lattice either by adding
        increments of magnetism or avalanching
        The time variable is used to make the movie.
        it specifies which frames should be saved. This is done with hexagonal
        nearest neighbours"""

        nn = np.zeros((self.N, self.N))  # Initialize nearest neighbours
        # Grid of nearest neighbours
        nn[1:-1, 1:-1] = self.lat_B[0:-2, 1:-1] + self.lat_B[1:-1, 0:-2] + self.lat_B[2:, 1:-1] +\
                         self.lat_B[1:-1, 2:] + self.lat_B[0:-2, 0:-2] + self.lat_B[2:, 2:]
        zk = self.lat_B - nn / 3 / self.D  # Grid of curvatures
        unst = np.where(zk > self.Z_c, 1, 0)  # Grid of 1 if unstable and 0 if stable
        # unstable points decrease and neighbours increase
        self.lat_C += -unst*self.Z_c * 3 * self.D / (3 * self.D + 1) +\
            np.roll(unst, 1, axis=0) * self.Z_c / (3 * self.D + 1) + \
            np.roll(unst, 1, axis=1) * self.Z_c / (3 * self.D + 1) + \
            np.roll(unst, -1, axis=0) * self.Z_c / (3 * self.D + 1) + \
            np.roll(unst, -1, axis=1) * self.Z_c / (3 * self.D + 1) + \
            np.roll(np.roll(unst, -1, axis=1), -1, axis=0) * self.Z_c / (3 * self.D + 1) + \
            np.roll(np.roll(unst, 1, axis=1), 1, axis=0) * self.Z_c / (3 * self.D + 1)
            # Energy dissipated in the process
        e = np.sum(3 * self.D / (3 * self.D + 1) * (2 * np.abs(zk[zk > self.Z_c]) / self.Z_c - 1) * self.Z_c ** 2)

        # If there is an avalanche
        if e > 0:
            if not self.avalanching:
                self.a_statistic_init(step_time)  # initialize the statistics
            if self.a_P[-1] < e/self.e0:  # update peak energy release
                self.a_P[-1] = e/self.e0
            self.a_E[-1] += e/self.e0  # Update total energy release
            # updates the lattice array
            self.lat_B[1:-1, 1:-1] += self.lat_C[1:-1, 1:-1]
            self.lat_C = np.zeros((self.N, self.N))

        # If there is no avalanche
        else:  # Otherwise, add increment to grid
            if self.avalanching:  # End of the avalanche
                self.avalanching = False
                self.a_T[-1][0] = step_time - self.a_T[-1][0]  # Ending time of the avalanche
            # Find random lattice element
            randx, randy = np.random.randint(1, high=self.N - 1), np.random.randint(1, high=self.N - 1)
            db = np.random.random() * np.abs(self.sigma1 - self.sigma2) + self.sigma1  # Random magnetism increment
            self.lat_B[randx][randy] += db  # Update lattice

        self.er.append(e / self.e0)  # Save the released energy
        self.el.append(np.sum(self.lat_B ** 2))  # save the lattice energy
        if self.images:  # Save images if we want them
            if not step_time % self.images:  # Save images with frequency 1/self.images
                list_holder = list(np.copy(self.lat_B))
                self.mat_history.append(list_holder)

    def loop(self, total_time, frequency, save=False, load=False, stats=False):
        """Run the avalanche simulation for total_Time iterations.
        print the loading bar at 1/frequency iterations. Prints the time
        it took to run at the end of the loop.
        if save, saves lat_B to 'save'
        if load, loads the numpy array 'load' as lat_B"""
        if load:
            self.lat_B = np.load(load)['lat_B']
        start = time.time()
        for i in range(int(total_time)):  # The actual loop
            if not i % int(frequency):  # Progress bar
                print(str(int(i / time_ * 100)) + '%')
            if self.nn_type == 'Normal':
                self.step(i)
            elif self.nn_type == 'Hex':
                self.step_hex(i)
            else:
                raise 'Nearest Neighbour type is not existent.'
        end = time.time()
        print('loop took '+str(round(end - start, 2)) + 's')
        # The stats about the last avalanche can be misleading
        avalanche1.a_E.pop()
        avalanche1.a_P.pop()
        avalanche1.a_T.pop()
        if save:  # Save the state of lat_B
            self.save_state(save)
        if stats:  # Save the stats of the avalanche
            self.save_stats(stats)



if __name__ == '__main__':
    # avalanche1 = Avalanche(2, 32, -0.2, 0.8, images=False)
    # time_ = int(avalanche1.sugg_time)
    # avalanche1.loop(time_, 100000, load = '/home/hlamarre/PycharmProjects/Avalanches/Saves/N32_SOC.npz',
    #                 stats = '/home/hlamarre/PycharmProjects/Avalanches/Saves/N32_stats.npz')
    # Met.plot_energies(avalanche1.el, avalanche1.er, time_, avalanche1.e0)
    # Met.plot_statistics(avalanche1.a_E, avalanche1.a_P, avalanche1.a_T)

    avalanche2 = Avalanche(2, 32, -0.2, 0.8, images=False, nn_type='Hex')
    time_ = int(avalanche2.sugg_time/4)
    avalanche2.loop(time_, 100000, stats = '/home/hlamarre/PycharmProjects/Avalanches/Saves/N32_hex_stats.npz')
    Met.plot_energies(avalanche2.el, avalanche2.er, time_, avalanche2.e0)
    Met.plot_statistics(avalanche2.a_E, avalanche2.a_P, avalanche2.a_T)