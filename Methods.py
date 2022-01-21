"""methods.py  Contains methods used to plot various aspects
of the avalanche.py class

Created 2021-09-07"""

__author__ = "Henri Lamarre"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from scipy.optimize import curve_fit
# matplotlib.use('nbAgg')


def make_movie(img):
    ''' Makes a movie mp4 with the supplied frames in img
    img is a list of 2d arrays to be displayed using plt.imshow'''
    frames = [] # for storing the generated images
    fig = plt.figure()
    for i in range(len(img)):
        frames.append([plt.imshow(img[i])])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)
    ani.save('avalanche.mp4')
    # plt.show()


def plot_energies(el, er, t, e0):
    ''' Plots the released energy (el) and lattice energy (er)
    as a function of time'''
    fig = plt.figure()
    ax1 = plt.subplot(211)
    ax1.plot(np.arange(t), np.array(er))
    ax1.set_ylabel('er')

    # share x only
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(np.arange(t), np.array(el)/e0)
    ax2.set_ylabel('el')
    plt.savefig('energies.png')
    plt.show()




def plot_statistics(E,P,T):
    """Plot the avalanche statistics"""
    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax1.scatter(P, E, marker='.')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('P/e0')
    ax1.set_ylabel('E/e0')

    # share y only
    ax2 = plt.subplot(122)
    ax2.scatter(T, P, marker='.')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_ylabel('P/e0')
    ax2.set_xlabel('T')
    plt.tight_layout()
    plt.savefig('statistics.png')
    plt.show()


def line(x, a, b):  # equation of a line
    return -a * x + b


def compute_slope(array, plot=False):
    """Computes the coefficient of the slope of the array. Makes a plot showing the fit
    returns the slope"""
    space = np.logspace(np.log(min(array)), np.log(max(array)), 70, base=np.e)  # Logspace for the fitting
    hist = np.histogram(array, bins=space[10:-15], density=True)  # density histogram
    for i in range(len(hist[0])):  # transforms 0 into small values for no errors
        if hist[0][i] < 1e-15:
            hist[0][i] = 1e-15
    x = (hist[1][:-1] + hist[1][1:]) / 2  # Average the bins sides for the fitting
    popt, pcov = curve_fit(line, np.log(x), np.log(hist[0]))  # fitting of the line
    if plot:  # if we want to see the fit
        plt.figure()
        plt.scatter(np.log(x), np.log(hist[0]))
        plt.plot(np.log(x), line(np.log(x), popt[0], popt[1]))
        plt.show()
    return popt[0]  # returns the slope


def slope_stats(path_, plot=False):
    """For a stat numpy file, compute the T,P,E slopes and prints them"""
    stats = np.load(path_)
    E = stats['a_E']  # Total energy release
    P = stats['a_P']  # Peak energy release
    T = stats['a_T']  # Time of avalanches
    del_index = []  #Remove the small and big avalanches for bias
    for i in range(len(E)):
        if np.log10(P[i]) < 1 or np.log10(T[i]) < 1.5 or np.log10(P[i]) > 3:
            del_index.append(i)
    E = np.delete(E, del_index)
    P = np.delete(P, del_index)
    T = np.delete(T, del_index)
    slope_E = compute_slope(E, plot)  # Compute the slopes
    slope_P = compute_slope(P, plot)
    slope_T = compute_slope(T, plot)
    print('E = {}, P = {}, T = {}'.format(slope_E, slope_P, slope_T))


def lattice_energy(path_):
    e_l = np.load(path_)['e_l']
    x_arr = np.linspace(min(e_l), max(e_l), 30)
    hist = np.histogram(e_l, bins=x_arr, density=True)
    plt.figure()
    plt.scatter(np.log(x_arr[:-1]), np.log(hist[0]))
    plt.xlabel('Lattice energy (log)')
    plt.ylabel('P(E) (log)')
    plt.show()

def loglog_plot(path_, key, num):
    quan = np.load(path_)[key]
    print(quan)
    x_arr = np.linspace(min(quan), max(quan), num)
    hist = np.histogram(quan, bins=x_arr, density=True)
    plt.figure()
    plt.scatter(np.log(x_arr[:-1]), np.log(hist[0]))
    plt.show()


if __name__ == '__main__':
    # slope_stats('/home/hlamarre/PycharmProjects/Avalanches/Saves/N32_hex_stats.npz', plot=True)
    # slope_stats('/home/hlamarre/PycharmProjects/Avalanches/Saves/N32_stats.npz', plot=True)
    # lattice_energy('/home/hlamarre/PycharmProjects/Avalanches/Saves/N10_Farhang_stats.npz')
    # loglog_plot('/home/hlamarre/PycharmProjects/Avalanches/Saves/N10_Farhang_stats.npz', 'a_e', 30)
    curvs = np.load('./Saves/N32_Farhang_curvs.npz')['curvs'][-100:]
    make_movie(curvs)