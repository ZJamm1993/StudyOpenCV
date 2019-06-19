from cv2 import cv2
import numpy as np

# Image Addition

# You can add two images by OpenCV function, cv2.add() or simply by numpy operation, res = mainImg + img2. Both images should be of same depth and type, or second image can just be a scalar value.

# Note

# There is a difference between OpenCV addition and Numpy addition. OpenCV addition is a saturated operation while Numpy addition is a modulo operation.

x = np.uint8([250])
y = np.uint8([10])
print cv2.add(x, y)
print x + y

# Image Blending

# This is also image addition, but different weights are given to images so that it gives a feeling of blending or transparency.

# mainImg = cv2.imread('resource/lena.jpg')
# mainImg = mainImg[0:100, 0:100]
# img2 = cv2.imread('resource/opencv_logo.png')
# img2 = img2[0:100, 0:100]

# dst = cv2.addWeighted(mainImg,0.7,img2,0.3,0)

# cv2.imshow('dst',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Bitwise Operations

# This includes bitwise AND, OR, NOT and XOR operations. They will be highly useful while extracting any part of the image (as we will see in coming chapters), defining and working with non-rectangular ROI etc. Below we will see an example on how to change a particular region of an image.

# I want to put OpenCV logo above an image. If I add two images, it will change color. If I blend it, I get an transparent effect. But I want it to be opaque. If it was a rectangular region, I could use ROI as we did in last chapter. But OpenCV logo is a not a rectangular shape. So you can do it with bitwise operations as below:

mainImg = cv2.imread('resource/lena.jpg')
img2 = cv2.imread('resource/opencv_logo.png')

# I want to put logo on top-left corner, So I create a ROI
rows, cols, channels = img2.shape
roi = mainImg[0:rows, 0:cols]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
bg_roi = cv2.bitwise_and(roi, roi, mask = mask_inv)

# Take only region of logo from logo image.
fg_img2 = cv2.bitwise_and(img2, img2, mask = mask)

# Put logo in ROI and modify the main image
dst = cv2.add(bg_roi, fg_img2)
mainImg[0:rows, 0:cols] = dst

cv2.imshow('res', mainImg)
cv2.waitKey(0)
cv2.destroyAllWindows()