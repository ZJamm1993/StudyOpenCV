from cv2 import cv2
import numpy as np

img = cv2.imread('resource/building.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create() # this SIFT algorithm is patented, I don't know how to use it
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp)

cv2.imshow('sift',img)
cv2.waitKey(0)
cv2.destroyAllWindows()