# Processor
imageProcessor = {
    "image": 'maps\jpegfloorplan.jpg',
    "image_out": 'maps\processed.jpg',
    "min_pixel_size": 0, # minimum size of particles we want to keep
    "noise_removal_threshold": 25, # Minimal area of blobs to be kept
    "corners_threshold": 0.1, # Threshold to allow corners. Higher removes more of the house
    "room_closing_max_length": 100, # Maximum line length to add to close off open doors
    "gap_in_wall_threshold": 500, # Minimum number of pixels to identify component as room instead of hole in the wall
    "binary_threshold": 128, # If pixel falls above/below threshold, make it white/black
    "x_threshold": 42, # number of x-axis pixels to close up the gap between 2 points on x-axis
    "y_threshold": 55, # number of y-axis pixels to close up the gap between 2 points on y-axis
}

#Cleaner
extractCoord = {
    "DBSCAN": True, # Activate DBSCAN clustering module
    "KMeans": False, # Activate KMeans clustering module
    "maxRange": 2000, # Max laser range
    "minRange": 24, # Min laser range
}

# Path
pathPlanner = {
    "AStar": False,
    "RRT": True,
    "RRTstepsize": 15,
    "Salesman": True
}

visual = {
    "output": 'maps/visualizedCandidates.jpg', # Output file path
    "thickness": 1, # Thickness of the FOV line
    "FOVColor": [0, 255, 0], # Colouring the scanned FOV
    "pointRadius": 2, # Size of the drawn scan candidates
}