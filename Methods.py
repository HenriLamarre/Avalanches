"""methods.py  Contains methods used to plot various aspects
of the avalanche.py class

Created 2021-09-07"""

__author__      = "Henri Lamarre"


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.animation as animation

def make_movie(img):
    ''' Makes a movie mp4 with the supplied frames in img
    img is a list of 2d arrays to be displayed using plt.imshow'''
    frames = [] # for storing the generated images
    fig = plt.figure()
    for i in range(len(img)):
        frames.append([plt.imshow(img[i])])

    ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True,
                                    repeat_delay=1000)
    ani.save('avalanche.mp4')
    # plt.show()

def plot_energies(el, er, t, e0):
    ''' Plots the released energy (el) and lattice energy (er)
    as a function of time'''
    fig = plt.figure()
    ax1 = plt.subplot(211)
    ax1.plot(np.arange(t), np.array(er))
    ax1.set_ylabel('er/e0')

    # share x only
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(np.arange(t), np.array(el)/e0)
    ax2.set_ylabel('el/e0')
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

    plt.show()
