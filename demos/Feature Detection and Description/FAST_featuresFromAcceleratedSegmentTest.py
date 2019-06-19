import numpy as np
from cv2 import cv2

img = cv2.imread('resource/block_test.png',0)

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = None
img2 = cv2.drawKeypoints(img, kp, img2, color=(255,0,0))

# Print all default params
# print "Threshold: ", fast.('threshold')
# print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
# print "neighborhood: ", fast.getInt('type')
# print "Total Keypoints with nonmaxSuppression: ", len(kp)

# Disable nonmaxSuppression
# fast.setBool('nonmaxSuppression',0)
fast = cv2.FastFeatureDetector_create(nonmaxSuppression = 0)
kp = fast.detect(img)

print "Total Keypoints without nonmaxSuppression: ", len(kp)

img3 = None
img3 = cv2.drawKeypoints(img, kp, img3, color=(255,0,0))

cv2.imshow('fast',img2)
cv2.imshow('fast nonmaxSuppression',img3)

cv2.waitKey(0)
cv2.destroyAllWindows()