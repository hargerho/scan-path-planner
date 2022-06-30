### https://pypi.org/project/tsp/
### Preliminary edit
import tsp
import pickle
import math
import numpy as np
import pandas as pd

### Create adjacency matrix for point 1 to all other points, point 2 to all other points etc
with open('SMARTray_dbs_nosubset(30-1333)_updatedmap.pickle', 'rb') as handle:
        scanning_list = pickle.load(handle)

scanning_list = list(scanning_list) #scanning_list = [(40,123), (40, 367), (40, 896), (500, 896), (423, 1068)]

scan_candidate_file = open("SMARTray_dbs_nosubset_correctedCoords(30-1333).pickle", "wb")
pickle.dump(coordSequence, scan_candidate_file)


class Salesman:
    def __init__(self):
        

    def getSequence(self, scanningList):
        n = len(self.scanning_list)
        G = np.empty((n, n), int)
        for i in range(len(self.scanning_list)):
            for j in range(len(self.scanning_list)):
                dist = math.sqrt(pow((self.scanning_list[i][0] - self.scanning_list[j][0]),2) + pow((self.scanning_list[i][1] - self.scanning_list[j][1]),2))
                G[i][j] = dist

        r = range(len(G))
        shortestPath = {(i,j) : G[i][j] for i in r for j in r}
        pathSequence = tsp.tsp(r, shortestPath)
        coordSequence = []

        for i in pathSequence[0]:
            coordSequence.append(self.scanning_list[i])

        # Reversing the coordinates x,y position in tuple
        coordSequence = list(map(lambda tup: tup[::-1], coordSequence))

        return coordSequence
