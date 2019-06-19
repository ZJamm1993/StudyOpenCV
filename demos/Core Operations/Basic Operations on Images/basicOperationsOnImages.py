# Goal

# Access pixel values and modify them
# Access image properties
# Setting Region of Image (ROI)
# Splitting and Merging images

# Almost all the operations in this section is mainly related to Numpy rather than OpenCV. A good knowledge of Numpy is required to write better optimized code with OpenCV.

# ( Examples will be shown in Python terminal since most of them are just single line codes )

from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load a color image
img = cv2.imread('resource/lena.jpg')

# Accessing and Modifying pixel values
px = img[10, 20]
print px

blue = img[100, 100, 0]
print blue

img[100, 100] = [0, 255, 0]
print img[100, 100]

# Above mentioned method is normally used for selecting a region of array, say first 5 rows and last 3 columns like that. For individual pixel access, Numpy array methods, array.item() and array.itemset() is considered to be better. But it always returns a scalar. So if you want to access all B,G,R values, you need to call array.item() separately for all.

# Better pixel accessing and editing method :
# red
print img.item(101, 110, 2)

img.itemset((101, 110, 2), 255)
print img.item(101, 110, 2)

# Accessing Image Properties
# Image properties include number of rows, columns and channels, type of image data, number of pixels etc.
print img.shape # It returns a tuple of number of rows, columns and channels (if image is color)
print img.size # Total number of pixels is accessed by img.size
print img.dtype # Image datatype is obtained by img.dtype

# Image ROI

# Sometimes, you will have to play with certain region of images. For eye detection in images, first perform face detection over the image until the face is found, then search within the face region for eyes. This approach improves accuracy (because eyes are always on faces :D ) and performance (because we search for a small area).

# ROI is again obtained using Numpy indexing. Here I am selecting the ball and copying it to another region in the image:
# img[y1:y2, x1:x2]
messi = cv2.imread('resource/messi.jpg')
ball = messi[230:270, 270:320]
img[80:120, 80:130] = ball

# Splitting and Merging Image Channels
# The B,G,R channels of an image can be split into their individual planes when needed. Then, the individual channels can be merged back together to form a BGR image again. This can be performed by:
b,g,r = cv2.split(img)
# cv2.split() is a costly operation (in terms of time), so only use it if necessary. Numpy indexing is much more efficient and should be used if possible.
img = cv2.merge((r,r,r))
# or
# b = img[:,:,0]
# Suppose, you want to make all the red pixels to zero, you need not split like this and put it equal to zero. You can simply use Numpy indexing which is faster.
# img[:,:,1] = 0

# show lena
plt.subplot(231),plt.imshow(img,'gray'),plt.title('LENA')

# Making Borders for Images (Padding)
RED = [255,0,0]

img1 = cv2.imread('resource/opencv_logo.png')

replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=RED)

# plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()
