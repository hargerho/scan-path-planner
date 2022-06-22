from concurrent.futures import process
from common import config

# Importing the module runners
from src.runner import scanRunner as scanRunner
# from src.runner import pathRunner as pathRunner

### Loading up the configs
def imageProcessor_loader():
    return config.imageProcessor

def extractCoord_loader():
    return config.extractCoord

def main():
    extractedCoordinates = scanRunner(
        processor=imageProcessor_loader(),
        cleaner=extractCoord_loader())

    # TODO run scanPlanner
    # TODO run pathPlanner

if __name__ == "__main__":
    main()