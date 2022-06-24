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

    def extracCoord(self):
        self.DataCleaner.distanceCalculator

    
