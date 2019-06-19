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

# 1. Data with Only One Feature

# Consider, you have a set of data with only one feature, ie one-dimensional. For eg, we can take our t-shirt problem where you use only height of people to decide the size of t-shirt.

# So we start by creating data and plot it in Matplotlib

import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

# 2. Data with Multiple Features

# In previous example, we took only height for t-shirt problem. Here, we will take both height and weight, ie two features.

# Remember, in previous case, we made our data to a single column vector. Each feature is arranged in a column, while each row corresponds to an input test sample.

# For example, in this case, we set a test data of size 50x2, which are heights and weights of 50 people. First column corresponds to height of all the 50 people and second column corresponds to their weights. First row contains two elements where first one is the height of first person and second one his weight. Similarly remaining rows corresponds to heights and weights of other people.

# create some random weight, height data
wei = 25
Z = np.random.randint(25, 50, (1, 2))
for i in range(10):
    rand = np.random.randint(wei, wei + 50, (25,2))
    Z = np.vstack((Z, rand))
    wei = wei + 20

# convert to np.float32
Z = np.float32(Z)

count = 10

# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(Z, count, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now separate the data, Note the flatten()
colors = ['r', 'g', 'b', 'm', 'c', 'r', 'g', 'b', 'm', 'c', 'r', 'g', 'b', 'm', 'c']
ravel = label.ravel()
for i in range(count):
    A = Z[ravel == i]
    plt.scatter(A[:,0],A[:,1],c = colors[i])
# Plot the data
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()