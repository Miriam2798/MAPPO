import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time


def transform(matrix, th):
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            if (matrix[row][col] > float(th)):
                matrix[row][col] = 1
            else:
                matrix[row][col] = 0
    return matrix


def main():
    print(r"""   _____      _____ ____________________________   
  /     \    /  _  \\______   \______   \_____  \  
 /  \ /  \  /  /_\  \|     ___/|     ___//   |   \ 
/    Y    \/    |    \    |    |    |   /    |    \
\____|__  /\____|__  /____|    |____|   \_______  /
        \/         \/                           \/ v0.1 """)
    time.sleep(1)
    matrix = np.loadtxt(open("map.csv", "rb"), delimiter=",", skiprows=1)
    th = input("Please define your Threshold: ")
    bordermatrix = transform(matrix, th)
    np.savetxt("bordermap.csv", bordermatrix, delimiter=",")
    print(bordermatrix)
    plt.imshow(bordermatrix, cmap=cm.binary)
    plt.show()


main()
