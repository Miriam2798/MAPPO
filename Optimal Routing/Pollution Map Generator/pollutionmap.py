import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time


# Heatmap Plotting function
def data_coord2view_coord(p, vlen, pmin, pmax):
    dp = pmax - pmin
    return (p - pmin) / dp * vlen


# Nearest Neighbour
def nearest_neighbours(xs, ys, reso, n_neighbours):
    im = np.zeros([reso, reso])
    extent = [np.min(xs), np.max(xs), np.min(ys), np.max(ys)]

    xv = data_coord2view_coord(xs, reso, extent[0], extent[1])
    yv = data_coord2view_coord(ys, reso, extent[2], extent[3])
    for x, y in itertools.product(range(reso), range(reso)):
        xp = (xv - x)
        yp = (yv - y)

        d = np.sqrt(xp**2 + yp**2)

        im[y][x] = 1 / np.sum(d[np.argpartition(d.ravel(),
                                                n_neighbours)[:n_neighbours]])

    return im, extent


# Exporting to JSON function
def export(im):

    # Export to csv file called map.csv
    np.savetxt("map.csv", im, delimiter=",")


def main():
    print(
        r"""__________      .__  .__          __  .__                   _____                 
\______   \____ |  | |  |  __ ___/  |_|__| ____   ____     /     \ _____  ______  
 |     ___/  _ \|  | |  | |  |  \   __\  |/  _ \ /    \   /  \ /  \\__  \ \____ \ 
 |    |  (  <_> )  |_|  |_|  |  /|  | |  (  <_> )   |  \ /    Y    \/ __ \|  |_> >
 |____|   \____/|____/____/____/ |__| |__|\____/|___|  / \____|__  (____  /   __/ 
                                                     \/          \/     \/|__|"""
    )
    time.sleep(1)
    generated_points = input(
        "Please specify the number of points generated (100 to 10000) default=1000: \n"
    )
    resolution = input(
        "Please specify the resolution (250 to 500) default=100: \n")
    nn = input(
        "Please, specify the Nearest Neighbour parameter (default=16): \n")

    # Generating random normal values from (0,1)
    x = np.random.randn(int(generated_points))
    x = x - min(x)
    # x = x / max(x)
    y = np.random.randn(int(generated_points))
    y = y - min(y)
    #y = y / max(y)

    # Store the values into im
    im, extent = nearest_neighbours(x, y, int(resolution), int(nn))

    # Normalize the values to (0,1)
    im = (im - np.min(im)) / np.ptp(im)

    # Plot im as a heatmap
    plt.imshow(im, origin='lower', extent=extent, cmap=cm.turbo)
    plt.show()

    # Export as a csv file
    export(im)