import math
import cv2
import os.path
from common import config

# Importing the module runners
from src.runner import pathRunner as pathRunner

### Loading up the configs
def imageProcessor_loader():
    return config.imageProcessor

def extractCoord_loader():
    return config.extractCoord

def path_loader():
    return config.pathPlanner

# Visualization functions
def visualize(img, scanning_dict):
    # Drawing out the FOV of the finalized scanned coordinates
    thickness = config.visual["thickness"]
    color = config.visual["FOVColor"]

    for scan_point, wall_list in scanning_dict.items():
        y_point, x_point = scan_point
        start_point = (x_point, y_point)
        for wall_points in wall_list:
            wall_points_y, wall_points_x = wall_points
            end_point = (wall_points_x, wall_points_y)

            # Draw Line
            cv2.line(img, start_point, end_point, color, thickness)
    
    # Visualizing the scan candidates
    for candidate in scanning_dict.keys():
        center_y, center_x = candidate
        radius = config.visual["pointRadius"]
        radius_sqr = radius*radius

        img[center_y][center_x] = 1

        for x in range(-radius, radius):
            hh = int(math.sqrt(radius_sqr - x*x))
            rx = center_x + x
            ph = center_y + hh
            
            for y in range(center_y - hh, ph):
                img[y][rx] = 255
    
    cv2.imwrite(config.visual["output"], img)


def main():
    pathPlanning = pathRunner.pathRunner(
        processor=imageProcessor_loader(),
        cleaner=extractCoord_loader(),
        path =path_loader())

    imgPath = 'maps/jpegfloorplan.jpg'

    # Start running the program
    scanningDict = pathPlanning.pathRun()

    if os.path.exist('maps/processed.jpg'):
        visualize(img=imgPath, scanning_dict=scanningDict)
    else:
        print("Cleaned image does not exist!")

if __name__ == "__main__":
    main()