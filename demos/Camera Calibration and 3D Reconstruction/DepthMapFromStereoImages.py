import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('resource/scene_l.jpg')
imgR = cv2.imread('resource/scene_r.jpg')
imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()