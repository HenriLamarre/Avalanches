"""avalanche.py  Contains a basic Avalanche class used 
for D-dimensional magnetic lattice avalanche SOC models

Created 2021-09-07"""
__author__ = "Henri Lamarre"

import numpy as np
import Methods as met


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
        db = 0.1
        self.Z_c = db / 0.0001 * 6 * 2 / self.N ** 2
        B_a = self.Z_c / 6 / 2 * self.N ** 2
        self.sugg_time = int(self.N ** 2 * B_a / db)


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

        for j in range_list_j:
            for k in range_list_k:
                NN = np.sum([self.lat_B[j - 1][k], self.lat_B[j + 1][k],
                             self.lat_B[j][k - 1], self.lat_B[j][k + 1]])
                Zk = self.lat_B[j][k] - NN / 2 / self.D
                if np.abs(Zk) > self.Z_c:
                    self.lat_C[j][k] = self.lat_C[j][k] - self.Z_c * 2 * self.D / self.s
                    self.lat_C[j - 1][k] = self.lat_C[j - 1][k] + self.Z_c / self.s
                    self.lat_C[j + 1][k] = self.lat_C[j + 1][k] + self.Z_c / self.s
                    self.lat_C[j][k - 1] = self.lat_C[j][k - 1] + self.Z_c / self.s
                    self.lat_C[j][k + 1] = self.lat_C[j][k + 1] + self.Z_c / self.s
                    g = 2 * self.D / self.s * (2 * np.abs(Zk) / self.Z_c - 1) * self.Z_c ** 2
                    e += g
        # If there is an avalanche
        if e > 0:
            # Rescale the mask of the avalanche
            (fnx,fny) = np.nonzero(self.lat_C)
            self.width_j[0] = min(fnx[fnx>0])
            self.width_j[1] = max(fnx[fny<self.N-1])
            self.width_k[0] = min(fny[fny>0])
            self.width_k[1] = max(fny[fny<self.N-1])
            print(self.width_k, self.width_j)

            # updates the lattice array
            self.lat_B[1: -1][1: -1] += self.lat_C[1: -1][1: -1]
            self.lat_C = np.zeros((self.N, self.N))
            # We will not check only the lattice increment in the next loop
            self.lattice_increment = False
        # Otherwise, add a magnetism increment
        else:
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


if __name__ == '__main__':
    avalanche1 = avalanche(2, 6, -0.2, 0.8, images=False)
    time = int(avalanche1.sugg_time)
    for i in range(time):
        if not i % 10000:
            print(i)
        avalanche1.step(i)
        # met.make_movie(avalanche1.mat_history)
    met.plot_energies(avalanche1.el, avalanche1.er, time, avalanche1.e0)
