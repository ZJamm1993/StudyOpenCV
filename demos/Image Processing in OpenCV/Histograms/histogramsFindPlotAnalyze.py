from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Histogram Calculation in OpenCV

# So now we use cv2.calcHist() function to find the histogram. Let’s familiarize with the function and its parameters :

# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])

# images : it is the source image of type uint8 or float32. it should be given in square brackets, ie, “[img]”.
# channels : it is also given in square brackets. It the index of channel for which we calculate histogram. For example, if input is grayscale image, its value is [0]. For color image, you can pass [0],[1] or [2] to calculate histogram of blue,green or red channel respectively.
# mask : mask image. To find histogram of full image, it is given as “None”. But if you want to find histogram of particular region of image, you have to create a mask image for that and give it as mask. (I will show an example later.)
# histSize : this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].
# ranges : this is our RANGE. Normally, it is [0,256].
# So let’s start with a sample image. Simply load an image in grayscale mode and find its full histogram.

img = cv2.imread('resource/lena.jpg')
# imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# hist = cv2.calcHist([imgray],[0],None,[256],[0,256])

# print hist

'''
Histogram Calculation in Numpy

Numpy also provides you a function, np.histogram(). So instead of calcHist() function, you can try below line :

hist,bins = np.histogram(img.ravel(),256,[0,256])

hist is same as we calculated before. But bins will have 257 elements, because Numpy calculates bins as 0-0.99, 1-1.99, 2-2.99 etc. So final range would be 255-255.99. To represent that, they also add 256 at end of bins. But we don’t need that 256. Upto 255 is sufficient.

See also

Numpy has another function, np.bincount() which is much faster than (around 10X) np.histogram(). So for one-dimensional histograms, you can better try that. Don’t forget to set minlength = 256 in np.bincount. For example, hist = np.bincount(img.ravel(),minlength=256)
Note

OpenCV function is more faster than (around 40X) than np.histogram(). So stick with OpenCV function.
'''

# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()

img = cv2.imread('resource/lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[50:150, 50:150] = 255
masked_img = cv2.bitwise_and(img,img,mask = mask)

# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img,'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])

plt.show()