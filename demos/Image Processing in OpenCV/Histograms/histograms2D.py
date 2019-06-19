# Introduction

# In the first article, we calculated and plotted one-dimensional histogram. It is called one-dimensional because we are taking only one feature into our consideration, ie grayscale intensity value of the pixel. But in two-dimensional histograms, you consider two features. Normally it is used for finding color histograms where two features are Hue & Saturation values of every pixel.

# There is a python sample in the official samples already for finding color histograms. We will try to understand how to create such a color histogram, and it will be useful in understanding further topics like Histogram Back-Projection.

from cv2 import cv2
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread('resource/building.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 2D Histogram in OpenCV

# It is quite simple and calculated using the same function, cv2.calcHist(). For color histograms, we need to convert the image from BGR to HSV. (Remember, for 1D histogram, we converted from BGR to Grayscale). For 2D histograms, its parameters will be modified as follows:

# channels = [0,1] because we need to process both H and S plane.
# bins = [180,256] 180 for H plane and 256 for S plane.
# range = [0,180,0,256] Hue value lies between 0 and 180 & Saturation lies between 0 and 256.
# Now check the code below:

hist = cv2.calcHist([hsv], [0,1], None, [180,256], [0,180,0,256])

'''
2D Histogram in Numpy

Numpy also provides a specific function for this : np.histogram2d(). (Remember, for 1D histogram we used np.histogram() ).

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('resource/home.jpg')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

hist, xbins, ybins = np.histogram2d(h.ravel(),s.ravel(),[180,256],[[0,180],[0,256]])
First argument is H plane, second one is the S plane, third is number of bins for each and fourth is their range.

Now we can check how to plot this color histogram.
'''

plt.imshow(hist,interpolation = 'nearest')
plt.show()
