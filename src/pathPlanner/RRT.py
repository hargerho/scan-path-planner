"""
Path planning with Rapidly-Exploring Random Trees (RRT)
Adapted from: Aakash(@nimrobotics)
web: nimrobotics.github.io
"""
import cv2
import numpy as np
import math
import random
import os

class Nodes:
    """Class to store the RRT graph"""
    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.parent_x = []
        self.parent_y = []


class RRTRun:
    def __init__(self, image, coordSequence, stepSize):
        self.stepSize = stepSize
        self.coordSequence = coordSequence
        self.IMAGE = image

    # check collision
    def collision(self,x1,y1,x2,y2,img):
        color=[]
        try:
            x = list(np.arange(x1,x2,(x2-x1)/100))
            y = list(((y2-y1)/(x2-x1))*(x-x1) + y1)
        except Warning as e:
                pass
        for i in range(len(x)):
            color.append(img[int(y[i]),int(x[i])])
        if (0 in color):
            return True #collision
        else:
            return False #no-collision

    # check the  collision with obstacle and trim
    def check_collision(self,x1,y1,x2,y2,img,end,stepSize):
        _,theta = self.dist_and_angle(x2,y2,x1,y1)
        x=x2 + stepSize*np.cos(theta)
        y=y2 + stepSize*np.sin(theta)

        hy,hx=img.shape
        if y<0 or y>hy or x<0 or x>hx:
            # Point out of image bound
            directCon = False
            nodeCon = False
        else:
            # check direct connection
            if self.collision(x,y,end[0],end[1],img):
                directCon = False
            else:
                directCon=True

            # check connection between two nodes
            if self.collision(x,y,x2,y2):
                nodeCon = False
            else:
                nodeCon = True

        return(x,y,directCon,nodeCon)

    # return dist and angle of new point and nearest node
    def dist_and_angle(self,x1,y1,x2,y2):
        dist = math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )
        angle = math.atan2(y2-y1, x2-x1)
        return(dist,angle)

    # return the neaerst node index
    def nearest_node(self,x,y,node_list):
        temp_dist=[]
        for i in range(len(node_list)):
            dist,_ = self.dist_and_angle(x,y,node_list[i].x,node_list[i].y)
            temp_dist.append(dist)
        return temp_dist.index(min(temp_dist))

    # generate a random point in the image space
    def rnd_point(self,h,l):
        new_y = random.randint(0, h)
        new_x = random.randint(0, l)
        return (new_x,new_y)


    def RRT(self,img, imgColor, start, end, stepSize):
        node_list = [0]
        h,l= img.shape 

        # insert the starting point in the node class
        # node_list = [0] list to store all the node points         
        node_list[0] = Nodes(start[0],start[1])
        node_list[0].parent_x.append(start[0])
        node_list[0].parent_y.append(start[1])

        # display start and end
        cv2.circle(imgColor, (start[0],start[1]), 5,(0,0,255),thickness=3, lineType=8)
        cv2.circle(imgColor, (end[0],end[1]), 5,(0,0,255),thickness=3, lineType=8)
        count = 0
        i=1
        pathFound = False
        while pathFound==False:
            nx,ny = self.rnd_point(h,l)

            nearest_ind = self.nearest_node(nx,ny,node_list)
            nearest_x = node_list[nearest_ind].x
            nearest_y = node_list[nearest_ind].y

            #check direct connection
            tx,ty,directCon,nodeCon = self.check_collision(nx,ny,nearest_x,nearest_y, img, end, stepSize)

            if directCon and nodeCon:
                node_list.append(i)
                node_list[i] = Nodes(tx,ty)
                node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
                node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
                node_list[i].parent_x.append(tx)
                node_list[i].parent_y.append(ty)

                cv2.circle(imgColor, (int(tx),int(ty)), 2,(0,0,255),thickness=1, lineType=8)
                cv2.line(imgColor, (int(tx),int(ty)), (int(node_list[nearest_ind].x),int(node_list[nearest_ind].y)), (0,255,0), thickness=1, lineType=8)
                cv2.line(imgColor, (int(tx),int(ty)), (end[0],end[1]), (255,0,0), thickness=2, lineType=8)

                # Path   found
                for j in range(len(node_list[i].parent_x)-1):
                    cv2.line(imgColor, (int(node_list[i].parent_x[j]),int(node_list[i].parent_y[j])), (int(node_list[i].parent_x[j+1]),int(node_list[i].parent_y[j+1])), (255,0,0), thickness=2, lineType=8)
                cv2.waitKey(1)
                cv2.imwrite("media/"+str(i)+".jpg",imgColor)
                cv2.imwrite("SMART_step15(media).jpg",imgColor)
                break

            elif nodeCon:
                # Nodes connected
                node_list.append(i)
                node_list[i] = Nodes(tx,ty)
                node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
                node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
                node_list[i].parent_x.append(tx)
                node_list[i].parent_y.append(ty)
                i=i+1

                # display
                cv2.circle(imgColor, (int(tx),int(ty)), 2,(0,0,255),thickness=3, lineType=8)
                cv2.line(imgColor, (int(tx),int(ty)), (int(node_list[nearest_ind].x),int(node_list[nearest_ind].y)), (0,255,0), thickness=1, lineType=8)
                cv2.imwrite("media/"+str(i)+".jpg",imgColor)
                cv2.imshow("sdc",imgColor)
                cv2.waitKey(1)
                count += 1
                continue

            else:
                continue
        return len(node_list), count, node_list
    
    def RRTRunner(self):
        # remove previously stored data
        try:
            os.system("rm -rf media")
        except:
            print("Dir already clean")
        os.mkdir("media")
        numNodes = 0

        # load grayscale image
        img = cv2.imread(self.IMAGE, 0)

        # load colored image
        imgColor = cv2.imread(self.IMAGE)

        for i in range(len(self.coordSequence) - 1):
            start = self.coordSequence[i]
            end = self.coordSequence[i+1]

            counter, count, nodeList = self.RRT(img, imgColor, start, end, self.stepSize)

            numNodes += counter
        
        return numNodes, count, nodeList


# if __name__ == '__main__':
#     # remove previously stored data
#     try:
#       os.system("rm -rf media")
#     except:
#       print("Dir already clean")
#     os.mkdir("media")
#     start_time = timeit.default_timer()
#     numNodes = 0

#     IMAGE_NAME = "RRT/smartMapOG.jpg"
#     img = cv2.imread(IMAGE_NAME,0) # load grayscale maze image
#     imgColor = cv2.imread(IMAGE_NAME) # load colored maze image

#     # # with open('ray_dbs_nosubset_corrected_coordinates_path_planning(24-2000).pickle', 'rb') as handle:
#     # #     scanning_list = pickle.load(handle)
#     scanning_list = [(1068, 423), (896, 500), (896, 40), (367, 40), (123, 40)]
#     for i in range(len(scanning_list)-1):
#         start = scanning_list[i]
#         end = scanning_list[i+1]

#     # # start = (561, 143) #(20,20) # starting coordinate
#     # # end = (385, 145) #(450,250) # target coordinate
#     # # end = tuple(args.stop) #(450,250) # target coordinate
#         stepSize = 15 # stepsize for RRT
#         node_list = [0] # list to store all the node points

#         # run the RRT algorithm 
#         counter, count = RRT(img, imgColor, start, end, stepSize)
#         numNodes += counter
        
#     stop_time = timeit.default_timer()
#     print("Time Taken ", round((stop_time - start_time), 4))
#     print("counter ", numNodes)

    

