'''
K-Means Clustering in OpenCV

Goal

Learn to use cv2.kmeans() function in OpenCV for data clustering
Understanding Parameters

Input parameters

samples : It should be of np.float32 data type, and each feature should be put in a single column.
nclusters(K) : Number of clusters required at end
criteria : It is the iteration termination criteria. When this criteria is satisfied, algorithm iteration stops. Actually, it should be a tuple of 3 parameters. They are ( type, max_iter, epsilon ):
3.a - type of termination criteria : It has 3 flags as below:
cv2.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached. cv2.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter. cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met.
3.b - max_iter - An integer specifying maximum number of iterations.
3.c - epsilon - Required accuracy
attempts : Flag to specify the number of times the algorithm is executed using different initial labellings. The algorithm returns the labels that yield the best compactness. This compactness is returned as output.
flags : This flag is used to specify how initial centers are taken. Normally two flags are used for this : cv2.KMEANS_PP_CENTERS and cv2.KMEANS_RANDOM_CENTERS.
Output parameters

compactness : It is the sum of squared distance from each point to their corresponding centers.
labels : This is the label array (same as ‘code’ in previous article) where each element marked ‘0’, ‘1’.....
centers : This is array of centers of clusters.
Now we will see how to apply K-Means algorithm with three examples.
'''

import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

'''
# 1. Data with Only One Feature
'''

# Consider, you have a set of data with only one feature, ie one-dimensional. For eg, we can take our t-shirt problem where you use only height of people to decide the size of t-shirt.

# So we start by creating data and plot it in Matplotlib

x = np.random.randint(25,100,25)
y = np.random.randint(175,255,25)
z = np.hstack((x,y))
z = z.reshape((50,1))
z = np.float32(z)
# plt.hist(z,256,[0,256]),plt.show()

# So we have ‘z’ which is an array of size 50, and values ranging from 0 to 255. I have reshaped ‘z’ to a column vector. It will be more useful when more than one features are present. Then I made data of np.float32 type.

# Now we apply the KMeans function. Before that we need to specify the criteria. My criteria is such that, whenever 10 iterations of algorithm is ran, or an accuracy of epsilon = 1.0 is reached, stop the algorithm and return the answer.

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS

# Apply KMeans
compactness,labels,centers = cv2.kmeans(z,2,None,criteria,10,flags)

# This gives us the compactness, labels and centers. In this case, I got centers as 60 and 207. Labels will have the same size as that of test data where each data will be labelled as ‘0’,‘1’,‘2’ etc. depending on their centroids. Now we split the data to different clusters depending on their labels.

A = z[labels==0]
B = z[labels==1]
# Now we plot A in Red color and B in Blue color and their centroids in Yellow color.

# Now plot 'A' in red, 'B' in blue, 'centers' in yellow
plt.hist(A,256,[0,256],color = 'r')
plt.hist(B,256,[0,256],color = 'b')
plt.hist(centers,32,[0,256],color = 'y')
plt.show()
