'''
Image Denoising in OpenCV

OpenCV provides four variations of this technique.

    cv2.fastNlMeansDenoising() - works with a single grayscale images
    cv2.fastNlMeansDenoisingColored() - works with a color image.
    cv2.fastNlMeansDenoisingMulti() - works with image sequence captured in short period of time (grayscale images)
    cv2.fastNlMeansDenoisingColoredMulti() - same as above, but for color images.
Common arguments are:
    h : parameter deciding filter strength. Higher h value removes noise better, but removes details of image also. (10 is ok)
    hForColorComponents : same as h, but for color images only. (normally same as h)
    templateWindowSize : should be odd. (recommended 7)
    searchWindowSize : should be odd. (recommended 21)
Please visit first link in additional resources for more details on these parameters.

We will demonstrate 2 and 3 here. Rest is left for you.
'''

# 1. cv2.fastNlMeansDenoisingColored()

# As mentioned above it is used to remove noise from color images. (Noise is expected to be gaussian). See the example below:

import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

img = cv2.imread('resource/noising_test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()