"""
ImageProcessor: Clean the annotated image and label each room with a unique color
Input: Floorplan Image
Output: room_coord and wall_dict
"""

import cv2
import numpy as np
import pandas as pd
import math

class ImageProcessor:
    def __init__(self, processor):
        self.IMAGE_NAME = processor["image"]
        self.IMAGE_OUT = processor["image_out"]
        
        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        # 0 yield best results
        self.min_pixel_size = processor["min_pixel_size"]
        self.NOISE_REMOVAL_THRESHOLD = processor["noise_removal_threshold"]
        self.CORNERS_THRESHOLD = processor["corners_threshold"]
        self.ROOM_CLOSING_MAX_LENGTH = processor["room_closing_max_length"]
        self.GAP_IN_WALL_THRESHOLD = processor["gap_in_wall_threshold"]
        self.BINARY_THRESHOLD = processor["binary_threshold"]
        self.X_THRESHOLD = processor["x_threshold"]
        self.Y_THRESHOLD = processor["y_threshold"]

    def remove_annotation(self):
        '''
        Source: https://stackoverflow.com/questions/54274610/separate-rooms-in-a-floor-plan-using-opencv
        Only works if the wall boundaries are coloured black
        '''
        # Convert image to grayscale
        im_gray = cv2.imread(self.IMAGE_NAME, cv2.IMREAD_GRAYSCALE)

        # Morphological Transform
        kernel = np.ones((5, 5), np.uint8)
        dilated_image = cv2.dilate(im_gray, kernel)

        return dilated_image

    def colour_rooms(self, dilated_image):
        '''
        Source: Source: https://stackoverflow.com/questions/54274610/separate-rooms-in-a-floor-plan-using-opencv?answertab=active#tab-top
        cornerHaris source: https://titanwolf.org/Network/Articles/Article?AID=e0121078-7654-4b48-8d03-6bdde54f1b58#gsc.tab=0
        '''

        # Converting the pixes to black and white
        dilated_image[dilated_image < self.BINARY_THRESHOLD] = 0
        dilated_image[dilated_image > self.BINARY_THRESHOLD] = 255

        # Convert type to uint8 for ~ operator to invert image
        img = dilated_image.astype(np.uint8)

        # Inverting the image using ~ operator (walls white)
        contours, hierarchy = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(img)
        for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.NOISE_REMOVAL_THRESHOLD:
                    cv2.fillPoly(mask, [contour], 255)
        
        # Inverting mask image
        img = ~mask

        # Get the corners in the floor plan
        # Return a 2D array (same size as input) of probabilites
        # Each position in array is confidence level of the neighbourhood that is centered at this / 
        # position being a corner
        dst = cv2.cornerHarris(img ,2,3,0.04)
        dst = cv2.dilate(dst,None)

        # Threshold for an optimal value, it may vary depending on the image.
        # Filter out low confidence corners
        # Corner marked if confidence higher than 1% of the highest confidence
        corners = dst > self.CORNERS_THRESHOLD * dst.max()

        # TODO edit here to close up the boundaries
        # Draw lines to close the rooms off by adding a line between corners on the same x or y coordinate
        # This gets some false positives
        # You could try to disallow drawing through other existing lines for example
        for y,row in enumerate(corners):
                # Get all whitespace (non-zero in array)
                x_same_y = np.argwhere(row)
                
                # zip function pairs up values in 2 lists of same index to form new tuple
                # [:-1] remove last value of  list
                # [1:] remove first value of list
                # x1 and x2 are strings for zip to work
                for x1, x2 in zip(x_same_y[:-1], x_same_y[1:]):
                    if x2[0] - x1[0] < self.X_THRESHOLD:
                        color = 0
                        cv2.line(img, (int(x1), int(y)), (int(x2), int(y)), color, 1)

        # .T is to transpose an array
        for x,col in enumerate(corners.T):
                y_same_x = np.argwhere(col)
                for y1, y2 in zip(y_same_x[:-1], y_same_x[1:]):
                    if y2[0] - y1[0] < self.Y_THRESHOLD:
                        color = 0
                        cv2.line(img, (int(x), int(y1)), (int(x), int(y2)), color, 1)

        # Filling the outer image black, inner free space remain white
        # Getting the contours of the inverted image (walls are white)
        contours, hierarchy = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get all the contour sizes
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        
        #Filter out the largest contour (exterior walls)
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        
        # Create a black masking
        mask = np.zeros_like(mask)

        # Fill the mask on the biggest contour
        cv2.fillPoly(mask, [biggest_contour], 255)
        img[mask == 0] = 0
        

        # Colouring the rooms
        ret, labels = cv2.connectedComponents(img)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        unique = np.unique(labels)
        rooms = {}
        key = 0
        for label in unique:
            component = labels == label
            if img[component].sum() == 0 or np.count_nonzero(component) < self.GAP_IN_WALL_THRESHOLD:
                color = 0
            else:
                color = np.random.randint(0, 255, size=3)

                # only takes in list of rgb, reject 0 (black)
                if type(color) != int:
                    color = color.tolist()
                    rooms[key] = color
                    key += 1
            
            img[component] = color

        # Writing the colored-coded rooms
        cv2.imwrite(self.IMAGE_OUT, img)
        print("Cleaned image safed")
                
        return img, rooms
    
    def roomCoordinates(self, img, rooms):
        # Reformat room_coord_list to store rgb of room and their coordinates
        # Filter out all black pixels
        # Source: https://stackoverflow.com/questions/27026866/convert-an-image-to-2d-array-in-python
        indices = np.dstack(np.indices(img.shape[:2]))
        map = np.concatenate((img, indices), axis=-1)
        room_coord_list = []
        black = np.array([0,0,0])
        for i in range(len(map)):
            for j in range(len(map[i])):
                point = map[i][j]
                rgb = point[:3]
                if np.all(rgb != black):
                    room_coord_list.append(point)

        # Good note when deciding row,col for y,x in the tuple. As long as you iterate properly, dosent matter if y,x or x,y
        # To standardize [row][col] == [y][x]
        # https://stackoverflow.com/questions/2203525/are-the-x-y-and-row-col-attributes-of-a-two-dimensional-array-backwards

        # Finding room coordinates
        rooms_coord = {}
        # Iterate through the rooms dict
        for room_num, tmp_room in rooms.items():
            tmp_list = []

            # Iterature through the room_coord_list without black pixels
            for i in range(len(room_coord_list)):

                # To store the found rgb identity of the room
                room_point = room_coord_list[i][:3]

                # Checking rbg of tmp_room and room_point
                if np.all(room_point == tmp_room):

                    # Store the coordinates only
                    # convert coord to tuple so that coordintes are immutable
                    tuple_coord = tuple(room_coord_list[i][3:].tolist())

                    # list of tuples
                    tmp_list.append(tuple_coord)

            rooms_coord[room_num] = tmp_list

        wall_dict = {}

        # Finding wall coordinates
        for room_number in rooms_coord:
            coord = rooms_coord.get(room_number)
            wall_list = []
            # Extract the x and y coordinates of the coloured_points/coloured pixels
            for coloured_points in coord:
                coloured_points_x, coloured_points_y = coloured_points

                # checking for black pixels surrounding the point
                black = np.array([0,0,0])
                northpoint = map[coloured_points_x][coloured_points_y + 1]
                northpoint_rgb = northpoint[:3]

                southpoint = map[coloured_points_x][coloured_points_y - 1]
                southpoint_rgb = southpoint[:3]

                westpoint = map[coloured_points_x - 1][coloured_points_y]
                westpoint_rgb = westpoint[:3]

                eastpoint = map[coloured_points_x + 1][coloured_points_y]
                eastpoint_rgb = eastpoint[:3]

                if np.all(northpoint_rgb == black):
                    tuple_point = tuple(northpoint[3:])
                    wall_list.append(tuple_point)

                if np.all(southpoint_rgb == black):
                    tuple_point = tuple(southpoint[3:])
                    wall_list.append(tuple_point)

                if np.all(westpoint_rgb == black):
                    tuple_point = tuple(westpoint[3:])
                    wall_list.append(tuple_point)

                if np.all(eastpoint_rgb == black):
                    tuple_point = tuple(eastpoint[3:])
                    wall_list.append(tuple_point)

            wall_dict[room_number] = wall_list

            return rooms_coord, wall_dict


# # Testing this module
# if __name__ == "__main__":
#     processor = ImageProcessor()
#     cleanedImg = processor.remove_annotation()
#     coloredImg, rooms = processor.colour_rooms(cleanedImg)
#     rooms_coord, wall_dict = processor.roomCoordinates(coloredImg,rooms)
#     cv2.imshow("out", coloredImg)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    
