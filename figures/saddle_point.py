import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def saddle_plot():
    ax = plt.figure().add_subplot(projection='3d')

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)

    x, y = np.meshgrid(x, y)

    z = x**2 - y**2

    # Plot the 3D surface
    ax.plot_surface(x, y, z,  edgecolor='royalblue', alpha=0.2)
    # ax.plot_surface(x, y, z, cmap=cm.Blues, alpha=0.9, zorder=0)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    plt.show()


if __name__ == '__main__':
    saddle_plot()
