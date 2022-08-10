import cv2
import numpy as np

# Importing the modules
from src.scanPlanner.ImageProcessor import ImageProcessor
from src.scanPlanner.extractCoord import DataCleaner, Clustering


class ScanRunner:
    def __init__(self, processor, cleaner):

        # Initalise the modules
        self.ImageProcessor = ImageProcessor(processor=processor)
        self.DataCleaner = DataCleaner(cleaner=cleaner)
    
    def imageProcesing(self):
        self.cleanedImg = self.ImageProcessor.remove_annotation()
        self.coloredImg, self.rooms = self.ImageProcessor.colour_rooms(self.cleanedImg)
        return self.ImageProcessor.roomCoordinates(self.coloredImg, self.rooms)

    def extracCoord(self, scanCandidates, wall):
        self.eligibleCandidates = self.DataCleaner.distanceCalculator(scan_candidates=scanCandidates, wall=wall)
        return self.DataCleaner.dataCleaner(eligible_candidates=self.eligibleCandidates)

    def scanRun(self):

        self.rooms_coord, self.wall_dict = self.imageProcesing()

        self.cleanDict = self.extracCoord(scanCandidates=self.rooms_coord, wall=self.wall_dict)

        return self.cleanDict