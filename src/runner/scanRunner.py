import cv2
import numpy as np

from src.scanPlanner.imageProcessor import imageProcessor
from src.scanPlanner.extractCoord import DataCleaner, Clustering


class ScanRunner:
    def __init__(self, processor, cleaner):
        self.imageProcessor = imageProcessor(processor=processor)
        self.DataCleaner = DataCleaner(cleaner=cleaner)
