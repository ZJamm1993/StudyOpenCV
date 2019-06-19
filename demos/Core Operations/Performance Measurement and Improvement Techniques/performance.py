from cv2 import cv2
import numpy as np

img1 = cv2.imread('resource/messi.jpg')
cv2.setUseOptimized(False)
e1 = cv2.getTickCount()
for i in xrange(1,100,2):
    img1 = cv2.medianBlur(img1,i)
e2 = cv2.getTickCount()
t = (e2 - e1)/cv2.getTickFrequency()
print t

loops = 1000000
#####

x = 5

e1 = cv2.getTickCount()
for i in xrange(loops):
    y = x**2
e2 = cv2.getTickCount()
t = (e2 - e1)/cv2.getTickFrequency()
print t

e1 = cv2.getTickCount()
for i in xrange(loops):
    y = x*x
e2 = cv2.getTickCount()
t = (e2 - e1)/cv2.getTickFrequency()
print t

# z = np.uint8([5])

# e1 = cv2.getTickCount()
# for i in xrange(loops):
#     y = z * z
# e2 = cv2.getTickCount()
# t = (e2 - e1)/cv2.getTickFrequency()
# print t

# e1 = cv2.getTickCount()
# for i in xrange(loops):
#     b = np.square(z)
# e2 = cv2.getTickCount()
# t = (e2 - e1)/cv2.getTickFrequency()
# print t

# Python scalar operations are faster than Numpy scalar operations. So for operations including one or two elements, Python scalar is better than Numpy arrays. Numpy takes advantage when size of array is a little bit bigger.