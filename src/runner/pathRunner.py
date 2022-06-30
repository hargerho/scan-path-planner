"""
Execute the path planner module and the downstream modules
"""

import cv2
import tsp
import math
import numpy as np
import pandas as pd

from src.runner.scanRunner import ScanRunner

# Import the path planning modules
from src.pathPlanner.RRT import RRT


class pathRunner:
    def __init__(self, processor, cleaner, path):
        
        # Initialize path planner variables
        self.AStar = path["AStar"]
        self.RRT = path["RRT"]
        self.RRTstepsize = path["RRTstepsize"]
        self.salesman = path["Salesman"]
        # self.img = read in the cleaned image from OS TODO

        # Initialize scanRunner
        self.ScanRunner = ScanRunner(processor=processor, cleaner=cleaner)

        # Initialize the path planning module
        self.RRT = RRT(img, coordSequence, self.RRTstepsize)

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

