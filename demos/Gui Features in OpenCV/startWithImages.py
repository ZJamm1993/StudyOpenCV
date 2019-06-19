import numpy as np
from cv2 import cv2 as cv
from matplotlib import pyplot as plt

image = cv.imread('resource/test.png', 0)

# # using cv show
# cv2.imshow('hello',image)
# # hold window
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# using pyplot show
# Color image loaded by OpenCV is in BGR mode. But Matplotlib displays in RGB mode. So color images will not be displayed correctly in Matplotlib if image is read with OpenCV. Please see the exercises for more details.

plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
plt.show()

# # save an image files
# cv2.imwrite('messigray.png',img)
