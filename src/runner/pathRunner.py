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
from src.pathPlanner.RRT import RRTRun


class pathRunner:
    def __init__(self, processor, cleaner, path):
        
        # Initialize path planner variables
        self.AStar_flag = path["AStar"]
        self.RRT_flag = path["RRT"]
        self.RRTstepsize = path["RRTstepsize"]
        self.salesman = path["Salesman"]
        self.img = cv2.imread(processor["image_out"])

        # Initialize scanRunner
        self.ScanRunner = ScanRunner(processor=processor, cleaner=cleaner)

        # Initialize the path planning module
        self.RRT = RRTRun(image=self.img, stepSize = self.RRTstepsize)

    def getSequence(self, scanning_list):
        n = len(scanning_list)
        G = np.empty((n, n), int)
        for i in range(len(scanning_list)):
            for j in range(len(scanning_list)):
                dist = math.sqrt(pow((scanning_list[i][0] - scanning_list[j][0]),2) + pow((scanning_list[i][1] - scanning_list[j][1]),2))
                G[i][j] = dist

        r = range(len(G))
        shortestPath = {(i,j) : G[i][j] for i in r for j in r}
        pathSequence = tsp.tsp(r, shortestPath)
        coordSequence = []

        for i in pathSequence[0]:
            coordSequence.append(scanning_list[i])

        # Reversing the coordinates x,y position in tuple
        coordSequence = list(map(lambda tup: tup[::-1], coordSequence))

        return coordSequence

    def pathRun(self):
        self.scanningDict = self.ScanRunner.scanRun()

        self.scanningList = list(self.scanningDict)

        self.coordSequence = self.getSequence(scanning_list=self.scanningList)

        # module toggle
        if self.RRT_flag is True:
            # Run the RRT module
            self.numNode, self.count, self.nodeList = self.RRT.RRTRunner(coordSequence=self.coordSequence)
        
        return self.scanningDict


