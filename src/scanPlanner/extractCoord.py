"""
scanPlanner: Get the coordinates on the floorplan with best coverage of the walls
Input: Room and wall coordinates
Output: Scan Point Coordinates
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import DBSCAN
from functools import reduce

class DataCleaner:
    def __init__(self, cleaner):

        # Equipment scanning range
        self.maxRange = cleaner["maxRange"]
        self.minRange = cleaner["minRange"]

        # initialte imageProcessor
        #self.imageProcessor = xxx

    # Calculating distance of scan_candidates to wall_points
    def distanceCalculator(self, scan_candidates, wall):
        eligible_candidates = {}

        for candidate in scan_candidates:
            y_point, x_point = candidate
            scanned_walls = []

            for wall_points in wall:
                wall_points_y, wall_points_x = wall_points # Key step
                dist = math.sqrt(pow((x_point - wall_points_x),2) + pow((y_point - wall_points_y),2))

                #trimble range 0.6m - 80m, 1 pixel = 0.025m, take 60m = 60/2400px (~2000px) for testing
                if dist <= self.maxRange and dist >=self.minRange:
                    scanned_walls.append(wall_points)
                    eligible_candidates[candidate] = scanned_walls
        return eligible_candidates

    # Data Cleaning
    def dataCleaner(self, eligible_candidates):
        checklist = []
        new_dict = {}

        # remove duplicates in each value
        eligible_candidates = {k:list(set(v)) for k, v in eligible_candidates.items()}

        # sorting dict by item length
        for k in sorted(eligible_candidates, key=lambda item: len(eligible_candidates[item]), reverse=True):
            for eachCoordinate in eligible_candidates[k]:
                if eachCoordinate not in checklist:
                    # if this key doesn't have any coordinates yet
                    if k in new_dict:
                        new_dict[k].append(eachCoordinate) # Append to exisitng value
                    else:
                        new_dict[k] = [eachCoordinate] # Create a new key-value with 1 coordinate in the list
                    checklist.append(eachCoordinate)
        return new_dict
    
class Clustering:
    def __init__(self, cleaner):

        # Selecting clusters
        self.DBSCAN = cleaner["DBSCAN"]
        self.Kmeans = cleaner["Kmeans"]

    def dbsClustering(self, new_dict):

        df = pd.DataFrame(list(new_dict.items()),columns = ['Scan_points','Wall_points'])
        
        # Splitting the column into x and y coordinates
        df[['x','y']] = pd.DataFrame(df['Scan_points'].tolist(), index=df.index)

        # Extract x and y column
        df2 = df[['x','y']]
        dbscan=DBSCAN()
        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(df2[['x','y']])
        distances, indices = nbrs.kneighbors(df2[['x','y']])

        distances = np.sort(distances, axis = 0)
        distances = distances[:,1]

        x = self.epsCalculator(distances)

        dbscan_opt=DBSCAN(eps=x,min_samples=4)
        dbscan_opt.fit(df2[['x','y']])
        df['DBSCAN_opt_labels']=dbscan_opt.labels_

        clustered_subset = df[['x','y','DBSCAN_opt_labels']]
        clustered_subset['Scan_points'] = list(zip(clustered_subset.x, clustered_subset.y))
        df['DBSCAN_opt_labels'] = df.Scan_points.map(clustered_subset.set_index('Scan_points')['DBSCAN_opt_labels'])
        df_max = df.sort_values(by='Wall_points', key=lambda x: x.str.len(), ascending=False, ignore_index=True)

        scanning_dict = {}
        cluster_list = []

        # Inserting first scan points and wall points set
        scanning_dict[df_max['Scan_points'][0]] = df_max['Wall_points'][0]
        cluster_list.append(df_max['DBSCAN_opt_labels'][0])

        # Appending all the best points from each cluster
        for idx, x in df_max['DBSCAN_opt_labels'].iteritems():
            if not x in cluster_list:
                scanning_dict[df_max['Scan_points'][idx]] = df_max['Wall_points'][idx]
                cluster_list.append(df_max['DBSCAN_opt_labels'][idx])

        # If doesent reach full coverage, continue to append scan points
        # flag = point_checker(wall, scanning_dict)
        # while self.point_checker(wall, scanning_dict):
        #     for idx, x in df_max['Scan_points'].iteritems():
        #         if not x in scanning_dict.keys():
        #             scanning_dict[df_max['Scan_points'][idx]] = df_max['Wall_points'][idx]
        
        return scanning_dict


    def epsCalculator(self, distances):
        maxSlope = 0
        eps = 1
        for i in range(1, len(distances)-1):
            x1 = i
            x2 = i+1
            y1 = distances[i]
            y2 = distances[i+1]
            slope = (y2-y1)/(x2-x1)
            if slope > maxSlope:
                maxSlope = slope
                eps = math.ceil(y1) 
        return eps

    # A flag to check if all the points have been scanned
    # def point_checker(self, wall_list, scanning_dict):
    #     flag = False
    #     wall_scanned = list(scanning_dict.values())
    #     wall_scanned = reduce(lambda x, y: x+y, wall_scanned)
    #     wall_scanned_set = set(wall_scanned)
    #     wall_set = set(wall_list)
    #     not_scanned = wall_set - wall_scanned_set

    #     if not_scanned:
    #         flag = True
        
    #     return flag

    # K - Means Clustering functions
    # Source: https://www.scikit-yb.org/en/latest/api/cluster/elbow.html?highlight=elbow
    # Source: https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/
    def kmeansClustering(self, new_dict):

        df = pd.DataFrame(list(new_dict.items()),columns = ['Scan_points','Wall_points'])
        
        # Splitting the column into x and y coordinates
        df[['x','y']] = pd.DataFrame(df['Scan_points'].tolist(), index=df.index)

        # Extract x and y column
        df2 = df[['x','y']]

        ## Plot Scanning points on x-y plane and use k-means++ clustering
        # Selecting thhe most optimal number of clusters
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df2)

        # Elbow Method for K means
        model = KMeans()

        # k is range of number of clusters.
        # Distortion score elbow for k-means: Distortion: It is calculated as the average of the squared distances 
        # from the cluster centers of the respective clusters. Typically, the Euclidean distance metric is used. 
        visualizer = KElbowVisualizer(model, k=(2,10), timings= False, locate_elbow= True)

        # Fit data to visualizer
        visualizer.fit(data_scaled)
        optimal = visualizer.elbow_value_

        centroid = self.kmeansCentroids(df2, optimal)
        clustered = self.kmeansFit(df2, centroid, optimal)

        clustered_subset = clustered[["x","y", "Cluster"]]
        clustered_subset['Scan_points'] = list(zip(clustered_subset.x, clustered_subset.y))
        df['Cluster'] = df.Scan_points.map(clustered_subset.set_index('Scan_points')['Cluster'])
        df_max = df.sort_values(by='Wall_points', key=lambda x: x.str.len(), ascending=False, ignore_index=True)
        
        scanning_dict = {}
        cluster_list = []

        # Inserting first scan points and wall points set
        scanning_dict[df_max['Scan_points'][0]] = df_max['Wall_points'][0]
        cluster_list.append(df_max['Cluster'][0])

        for idx, x in df_max['Cluster'].iteritems():
            if not x in cluster_list:
                scanning_dict[df_max['Scan_points'][idx]] = df_max['Wall_points'][idx]
                cluster_list.append(df_max['Cluster'][idx])   
        
        return scanning_dict

    def kmeansCentroids(self, X1, k):
        centroids = X1.sample(k, random_state = 1)
        i = 1
        dist = []
        while i != k:
            max_dist = [0,0]
            # Go through the centroids
            for index, row in centroids.iterrows():

                # calculate distance of every centroid with every other data point 
                d = np.sqrt((X1["x"] - row["x"])**2 +(X1["y"] - row["y"])**2)

                # check which centroid has a max distance with another point
                if max(d) > max(max_dist):
                    max_dist = d

            X1 = pd.concat([X1, max_dist], axis = 1)
            idx = X1.iloc[:,i+1].idxmax()
            max_coor = pd.DataFrame(X1.iloc[idx][["x", "y"]]).T
            centroids = pd.concat([centroids,max_coor])
            X1 = X1.drop(idx)
            i+=1
        return centroids

    def kmeansFit(self,X,centroids, n):
        # Get a copy of the original data
        X_data = X
        
        diff = 1
        j=0

        while(diff!=0):

            # Creating a copy of the original dataframe
            i=1

            # Iterate over each centroid point 
            for index1,row_c in centroids.iterrows():
                ED=[]

                # Iterate over each data point
                for index2,row_d in X_data.iterrows():

                    #calculate distance between current point and centroid
                    d1=(row_c["x"]-row_d["x"])**2
                    d2=(row_c["y"]-row_d["y"])**2
                    d=np.sqrt(d1+d2)

                    #append distance in a list 'ED'
                    ED.append(d)

                #append distace for a centroid in original data frame
                X[i]=ED
                i=i+1

            C=[]
            for index,row in X.iterrows():

                #get distance from centroid of current data point
                min_dist=row[1]
                pos=1

                #loop to locate the closest centroid to current point
                for i in range(n):

                    #if current distance is greater than that of other centroids
                    if row[i+1] < min_dist:

                        #the smaller distance becomes the minimum distance 
                        min_dist = row[i+1]
                        pos=i+1
                C.append(pos)

            #assigning the closest cluster to each data point
            X["Cluster"]=C

            #grouping each cluster by their mean value to create new centroids
            centroids_new = X.groupby(["Cluster"]).mean()[["y","x"]]
            if j == 0:
                diff=1
                j=j+1

            else:
                #check if there is a difference between old and new centroids
                diff = (centroids_new['y'] - centroids['y']).sum() + (centroids_new['x'] - centroids['x']).sum()
                print(diff.sum())

            centroids = X.groupby(["Cluster"]).mean()[["y","x"]]
            
        return X
