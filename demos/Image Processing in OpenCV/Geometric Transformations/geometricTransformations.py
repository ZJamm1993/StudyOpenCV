from cv2 import cv2
import numpy as np

# OpenCV provides two transformation functions, cv2.warpAffine and cv2.warpPerspective, with which you can have all kinds of transformations. cv2.warpAffine takes a 2x3 transformation matrix while cv2.warpPerspective takes a 3x3 transformation matrix as input.

img = cv2.imread('resource/lena.jpg')
cv2.imshow('origin', img)
rows, cols, dee = img.shape # like (200, 200, 3)

# scale
res = cv2.resize(img, None, fx = 2, fy = 1.5, interpolation = cv2.INTER_LINEAR)
cv2.imshow('scale', res)

# translation
tx = 100
ty = 50
m = np.float32([[1,0,tx],[0,1,ty]])
res = cv2.warpAffine(img, m, (cols, rows)) 
# Third argument of the cv2.warpAffine() function is the size of the output image, which should be in the form of (width, height). Remember width = number of columns, and height = number of rows.
cv2.imshow('trans', res)

# rotation
m = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
res = cv2.warpAffine(img, m, (cols,rows))
cv2.imshow('rotat', res)

# affine trans
ps1 = np.float32([[50, 100], [200, 100], [50, 200]])
ps2 = np.float32([[10, 10], [280, 20], [150, 150]])
m = cv2.getAffineTransform(ps1, ps2)
res = cv2.warpAffine(cv2.imread('resource/drawing.png'), m, (300,400))
cv2.imshow('affine', res)

# Perspective
ps1 = np.float32([[53, 58], [355, 51], [26,371], [379,379]])
ps2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
m = cv2.getPerspectiveTransform(ps1, ps2)
res = cv2.warpPerspective(cv2.imread('resource/sudo.png'), m, (400, 400))
cv2.imshow('perspective', res)

# hold window
cv2.waitKey(0)
cv2.destroyAllWindows()