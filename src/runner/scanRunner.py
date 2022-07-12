import cv2
import numpy as np

# Importing the modules
from src.scanPlanner.imageProcessor import imageProcessor
from src.scanPlanner.extractCoord import DataCleaner, Clustering


class ScanRunner:
    def __init__(self, processor, cleaner):

        # Initalise the modules
        self.imageProcessor = imageProcessor(processor=processor)
        self.DataCleaner = DataCleaner(cleaner=cleaner)
    
    def imageProcesing(self):
        self.cleanedImg = self.imageProcessor.remove_annotation()
        self.coloredImg, self.rooms = self.imageProcessor.colour_rooms(self.cleanedImg)
        return self.imageProcessor.roomCoordinates(self.coloredImg, self.rooms)

    def extracCoord(self, scanCandidates, wall):
        self.eligibleCandidates = self.DataCleaner.distanceCalculator(scan_candidates=scanCandidates, wall=wall)
        return self.DataCleaner.dataCleaner(eligible_candidates=self.eligibleCandidates)

    def ScanRun(self):

        self.rooms_coord, self.wall_dict = self.imageProcesing()

        self.cleanDict = self.extracCoord(scanCandidates=self.rooms_coord, wall=self.wall_dict)

        return self.cleanDict