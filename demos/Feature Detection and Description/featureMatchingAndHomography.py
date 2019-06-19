

import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('resource/box.jpg',0)          # queryImage
img2 = cv2.imread('resource/box_in_scene.jpg',0) # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher_create()
matches = bf.knnMatch(des1,des2,2)
good = []
for m,n in matches:
    if m.distance < 0.9*n.distance:
        good.append([m])

# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags = 2)
matchesMask = None
if len(good)>MIN_MATCH_COUNT:
    # what is it ?
    src_pts = np.float_([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float_([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float_([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None
    
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,(0,255,0),None,matchesMask,2)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None)
cv2.imshow('match knn',img3)

cv2.waitKey(0)
cv2.destroyAllWindows()