# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 22:29:19 2020

#@author Akshat Chauhan
"""

######################################################################
#The codes are based on Python3 
######################################################################
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, find
from itertools import combinations
from scipy.spatial import distance
import copy
# k-means clustering and K-medoid; 

def K_Means(x,cno=3,dim=2):
    
    m = x.shape[1]
    
    np.random.seed(int(time.time()))
    c = 6 * np.random.rand(dim, cno)
    c_old = copy.deepcopy(c) + 10
    i = 1
    # check whether the cluster centers still change
    tic = time.time()
    while np.linalg.norm(c - c_old, ord = 'fro') > 1e-6:
        print("--iteration %d \n" % i)
    
        #record previous c;  
        c_old = copy.deepcopy(c)
    
        # Assign data points to current cluster;
        # Squared norm of cluster centers.
        cnorm2 = np.sum(np.power(c,2), axis = 0)
        tmpdiff = 2 * np.dot(x.T,c) - cnorm2
        labels = np.argmax(tmpdiff,axis = 1)
    
        # Update data assignment matrix;
        # The assignment matrix is a sparse matrix,
        # with size m x cno. Only one 1 per row.
        P = csc_matrix( (np.ones(m) ,(np.arange(0,m,1), labels)), shape=(m, cno) )

        # adjust the cluster centers according to current assignment; 
        obj = 0
        for k in range(0,cno):
            idx = find(P[:,k])[0]
            nopoints = idx.shape[0]
            if nopoints == 0:
                # a center has never been assigned a data point; 
                # re-initialize the center; 
                c[:,k] = np.random.rand(dim,1)[:,0]
            else:
                # equivalent to sum(x(:,idx), 2) ./ nopoints;   
                c[:,k] = ((P[:,k].T.dot(x.T)).T / float(nopoints))[:,0] 

            obj = obj + np.sum(np.sum( np.power(x[:,idx] - c[:,k].T.reshape(dim,1), 2) ))
        print ("Objective Function is .......{}".format(obj))  

        i = i + 1

    toc = time.time()

    print('Elapsed time is %f seconds \n' % float(toc - tic))
    print('obj =', obj)
    
    return c.T,labels.T


def K_Medoids(x,cno=3,dim=2):
    
    m = x.shape[1]
    
    distance_measure=['euclidean','minkowski','correlation','chebyshev','sqeuclidean']
    
    # Please change below assignment to run for different distances
    
    d_m=distance_measure[0]
    
    np.random.seed(int(time.time()))
    #initialising Cluster center from a given data points in such manner 
    # that outliers or extreme points are never chosen as medoids..it reduces time complexity as well
    
    
    
    # We can choose either way of initializing the medoid.....
    
    # Option 2
    centroid=np.sum(x,axis=1)/x.shape[1] # calculating centroid 
    d=distance.cdist(x.T, centroid.reshape(1,dim), metric=d_m)
    n_nearest_points=np.argsort(d,axis=0)[:cno] # finding k nearest points to centroid 
    c=x[:,n_nearest_points[:,0]].T # k nearest points become initial medoids
    
    
    # Option 3
    c=x[:,0:cno].T # first k data points as centers
    
    
    # Option 1
    #c=x[:,np.random.randint(0, x.shape[1],size=[1,cno])].reshape(dim,cno).T # Choosing random initial center

 


    i = 1
    #calculating over all initial cost 

    dist_euclidean=distance.cdist(x.T, c, metric=d_m)
    label=np.argmin(dist_euclidean,axis = 1)
    obj=np.sum(np.amin(dist_euclidean,axis = 1))
    obj_new=0
    
    print("initial objective function.....{}\n\n".format(obj))

    # check whether the cost increases
    tic = time.time()
    #while np.linalg.norm(c - c_old, ord = 'fro') > 1e-6:
    while  (abs(obj-obj_new)>1e-6):
        print("--iteration %d \n" % i)
        #print("Old objective function.....{}".format(obj))
    
        obj=copy.deepcopy(obj_new)

        # Calculating distance 
   
        # Updating data assignment matrix;

        P = csc_matrix( (np.ones(m) ,(np.arange(0,m,1), label)), shape=(m, cno) )

        c_new=copy.deepcopy(c) # create new copy of medoids

        for k in range(0,cno): # dealing with one cluster at a time
            idx = find(P[:,k])[0]
            #plt.plot(x[0, idx], x[1, idx], cstr[k])
            y=x[:,idx] # intermittent data points assigned to a cluster

            # s here determines how many points within cluster which are closer to cluster centroid, needs to be examined for minimum total distance
            if (y.shape[1]>=50):
                s=20
            elif (y.shape[1]>=10 and y.shape[1]<50):
                s=15  
            elif (y.shape[1]<=2):
                s=1
            elif (y.shape[1]>2 and y.shape[1]<10):
                s=y.shape[1]

            if (s>1):
                o=np.sum(y,axis=1)/y.shape[1] # finding centroid of intermittent cluster
                d1=distance.cdist(y.T, o.reshape(1,dim), metric=d_m)
                dis1=np.sum(d1,axis=1)
                ci1=np.argsort(dis1,axis=0)[:][:s] # finding index of points which need to be checked for min distance within a cluster

                intr=y[:,ci1] # intermitten candidates for medoid of a cluster
                
                
            # We will calculate the shortest sum of distance from s points to entire cluster
                d2=distance.cdist(y.T, intr.T, metric=d_m)
                dis2=np.sum(d2,axis=1)
                ci2=np.argsort(dis2,axis=0)[:][0]
                c_new[k,:]=y[:,ci2].T
                c[k,:]=c_new[k,:]
            
                        
            
        dist_euclidean3=distance.cdist(x.T, c, metric=d_m)
        obj_new=np.sum(np.amin(dist_euclidean3,axis = 1)) # find new cost
        
        label=np.argmin(dist_euclidean3,axis = 1) # assigning new lables

        print ("New Objective Function is .......{}".format(obj_new))  
  
        
        i = i + 1

    toc = time.time()
    print('Elapsed time is %f seconds \n' % float(toc - tic))
    print('obj =', obj)
    
    return c,label.T

#####################################################################

from PIL import Image
from matplotlib.pyplot import imshow
#%matplotlib inline
def read_img(path):
#'''Read image and store it as an array, given the image path. Returns the 3 dimensional image array'''
    img = Image.open(path)
    img_arr = np.array(img, dtype='int32')
    img.close()
    return img_arr


def display_image(arr):
#display the image input : 3 dimensional array
    arr = arr.astype(dtype='uint8')
    img = Image.fromarray(arr, 'RGB')
    imshow(np.asarray(img))
    
    
    
    
    
    
    
    
####### Following section for evaluators to comment or uncomment based on requirement#############

# Please comment and uncomment based on requirement to load picture
    
#img_arr = read_img("football.bmp")
#img_arr=read_img("beach.bmp")
img_arr=read_img("MyCam.jpg")

r, c, l = img_arr.shape
img_reshaped = np.reshape(img_arr, (r*c, l), order="C")
img_reshaped.shape

# Please comment and uncomment based on requirement to run either K-Means or K-Medoids

# cno is cluster numbers 


#centers,labels=K_Medoids(img_reshaped.T,cno=20,dim=3)
centers,labels=K_Means(img_reshaped.T,cno=32,dim=3)
img_clustered = np.array([centers[i] for i in labels])
r, c, l = img_arr.shape
img_disp = np.reshape(img_clustered, (r, c, l), order="C")
display_image(img_disp)
