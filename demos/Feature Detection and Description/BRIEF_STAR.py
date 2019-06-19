import numpy as np
from cv2 import cv2

img = cv2.imread('resource/block_test.png',0)

# Initiate STAR detector
star = cv2.Feature2D.c

# Initiate BRIEF extractor
brief = cv2.DescriptorExtractor_create("BRIEF")

# find the keypoints with STAR
kp = star.detect(img,None)

# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)

print brief.getInt('bytes')
print des.shape