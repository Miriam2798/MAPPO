import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

# Transforms the matrix values of map.csv into 0 or 1 depending on the threshold value th
def transform(matrix, th):
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            matrix[row][col] = 1 if (matrix[row][col] > float(th)) else 0
    return matrix


def main():
    print(
        r"""___________                              _____                         _____                 
\__    ___/___________    ____   _______/ ____\___________  _____     /     \ _____  ______  
  |    |  \_  __ \__  \  /    \ /  ___/\   __\/  _ \_  __ \/     \   /  \ /  \\__  \ \____ \ 
  |    |   |  | \// __ \|   |  \\___ \  |  | (  <_> )  | \/  Y Y  \ /    Y    \/ __ \|  |_> >
  |____|   |__|  (____  /___|  /____  > |__|  \____/|__|  |__|_|  / \____|__  (____  /   __/ 
                      \/     \/     \/                          \/          \/     \/|__|"""
    )
    time.sleep(1)
    matrix = np.loadtxt(open("map.csv", "rb"), delimiter=",", skiprows=1)
    th = input("Please define your Threshold: ")
    bordermatrix = transform(matrix, th)
    np.savetxt("bordermap.csv", bordermatrix, delimiter=",")
    plt.imshow(bordermatrix, cmap=cm.binary)
    plt.show()