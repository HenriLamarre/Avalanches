import numpy as np 
import copy
import matplotlib.pyplot as plt
import avalancher2 as av

def e_total(lattice):
    """ Returns the total energy of a specified lattice """
    return np.sum(np.multiply(lattice[1:-1, 1:-1], lattice[1:-1, 1:-1]))

def curvature(lattice, nn=False):
    curv = np.zeros((32,32))  # curvature of the lattice initialization
    curv[1:-1, 1:-1] = lattice[1:-1, 1:-1] - 1 / 4 * (lattice[1:-1, 0:-2] + lattice[1:-1, 2:] +
                        lattice[0:-2, 1:-1] + lattice[2:, 1:-1])  # curvature computation
    return curv

def distribution(x, rs, a, Zc, grid, r, s, directions):
    new_grid = copy.deepcopy(grid)
    new_grid[r, s] -= 4 / 5*Zc
    new_grid[directions[0][0], directions[0][1]] += 4 / 5 * rs[0] / (x + a)*Zc
    new_grid[directions[1][0], directions[1][1]] += 4 / 5 * rs[2] / (x + a)*Zc
    new_grid[directions[2][0], directions[2][1]] += 4 / 5 * x / (x + a)*Zc
    new_grid[directions[3][0], directions[3][1]] += 4 / 5 * rs[1] / (x + a)*Zc
    return e_total(grid) - e_total(new_grid)

def solver(r,s, grid, curv, shuffle = False, plot=True):
    directions = np.array([[r-1,s],[r, s+1],[r, s-1],[r+1,s]])
    if shuffle:
        np.random.shuffle(directions)
    rs = np.random.uniform(0, 1, size=(3, 1))[:,0]
    Zc = 1
    a = np.sum(rs)
    x_space = np.linspace(-10,10, 1000)
    y_space = []
    for x in x_space:
        y_space.append(distribution(x, rs, a, Zc, grid, r ,s, directions))
    if np.max(y_space)>0:
        converged = True

    theta = rs[0]*grid[directions[0][0],directions[0][1]] + rs[1]*grid[directions[3][0],directions[3][1]]+\
    rs[2]*grid[directions[1][0],directions[1][1]]-a*grid[directions[2][0],directions[2][1]]
    phi = rs[0]**2+rs[1]**2+rs[2]**2
    C = 4*Zc/5
    x = (C*(phi)-theta*a)/(theta+C*a)
    ddx = 4*C/(x+a)**3*(theta-3*C/2/(x+a)*(phi-x*a) - C*a/2)

    if plot:
        fig, ax = plt.subplots(1,2)
        ax[0].plot(x_space, y_space)
        ax[0].vlines(x, min(y_space), max(y_space), color='black')
        ax[0].set_ylim(-10,10)
        ax[0].legend(['Energy', 'Optimal'])
        ax[0].set_xlabel('x value')
        ax[0].set_ylabel(r'energy $\sum A^2$')

        ax[1].matshow(curv[r-3:r+4, s-3:s+4])
        # for (i, j), z in np.ndenumerate(curv[5-3:5+4, 24-3:24+4]):
        #     ax[1].text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
        ax[1].arrow(3,3,0,-0.5, width = 0.1)
        ax[1].arrow(3,3,0.5,0, width = 0.1)
        ax[1].arrow(3,3,0,0.5, width = 0.1)
        ax[1].arrow(3,3,-0.5,0, width = 0.1)
        ax[1].text(2*(directions[2][0] - r) + 3, 2*(directions[2][1] - s) + 3, 'x = '+str(round(x, 2)), ha='center', va='center', color='white', size = 'x-large')
        ax[1].text(2*(directions[1][0] - r) + 3, 2*(directions[1][1] - s) + 3, 'r3 = '+str(round(rs[2], 2)), ha='center', va='center', color='white', size = 'x-large')
        ax[1].text(2*(directions[0][0] - r) + 3, 2*(directions[0][1] - s) + 3, 'r1 = '+str(round(rs[0], 2)), ha='center', va='center', color='white', size = 'x-large')
        ax[1].text(2*(directions[3][0] - r) + 3, 2*(directions[3][1] - s) + 3, 'r2 = '+str(round(rs[1], 2)), ha='center', va='center', color='white', size = 'x-large')
        plt.show()
    print('The maximal energy release for this system is '+str(max(y_space)))
    print('The double derivative is '+str(round(ddx,2)))

def curvature_tresh(latt):
    index_list = []
    for i in range(len(latt)):
        for j in range(len(latt)):
            if latt[i,j]>1:
                index_list.append([i,j])
    return index_list

def avalanche_sample(t_):
    avalanche1 = av.Avalanche(2, 32)
    avalanche1.name = 'avalancher'
    avalanche1.loop(t_, load = './', saving_lattice=False, progress_bar=False)
    return [avalanche1.energy_disp, avalanche1.energy_lat]