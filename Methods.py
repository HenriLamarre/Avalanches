"""methods.py  Contains methods used to plot various aspects
of the avalanche.py class

Created 2021-09-07"""

__author__      = "Henri Lamarre"


import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
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
    plt.plot(np.arange(t), np.array(er)/e0)

    # share x only
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(np.arange(t), np.array(el)/e0)
    plt.savefig('energies.png')
    plt.show()
