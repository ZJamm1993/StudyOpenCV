import numpy as np 
from cv2 import cv2


img = cv2.imread('resource/hand.jpg')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt = contours[0]

# Aspect Ratio
# It is the ratio of width to height of bounding rect of the object.
x,y,w,h = cv2.boundingRect(cnt)
aspect_ratio = float(w)/h
print 'aspect_ratio = '+str(aspect_ratio)

# Extent
# Extent is the ratio of contour area to bounding rectangle area.
area = cv2.contourArea(cnt)
rect_area = w*h
extent = float(area)/rect_area
print 'extent = '+str(extent)

# Solidity
# Solidity is the ratio of contour area to its convex hull area.
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
solidity = float(area)/hull_area
print 'solidity = '+str(solidity)

# Orientation
# Orientation is the angle at which object is directed. Following method also gives the Major Axis and Minor Axis lengths.
(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
print 'orientation = '+str(angle)

# Mask and Pixel Points
# In some cases, we may need all the points which comprises that object. It can be done as follows:
# Here, two methods, one using Numpy functions, next one using OpenCV function (last commented line) are given to do the same. Results are also same, but with a slight difference. Numpy gives coordinates in (row, column) format, while OpenCV gives coordinates in (x,y) format. So basically the answers will be interchanged. Note that, row = x and column = y.
mask = np.zeros(imgray.shape,np.uint8)
cv2.drawContours(mask,[cnt],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))
#pixelpoints = cv2.findNonZero(mask)
print 'pixelpoints = '+str(pixelpoints)

# Maximum Value, Minimum Value and their locations
# We can find these parameters using a mask image.
minMaxLoc = cv2.minMaxLoc(imgray, mask = mask)
print 'min,max,minloc,maxloc = '+str(minMaxLoc)

# Mean Color or Mean Intensity
# Here, we can find the average color of an object. Or it can be average intensity of the object in grayscale mode. We again use the same mask to do it.
mean_val = cv2.mean(img,mask = mask)
print 'mean_val = '+str(mean_val)

# Extreme Points
# Extreme Points means topmost, bottommost, rightmost and leftmost points of the object.
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
print 'left,right,top,bottom = '+str(leftmost)+str(rightmost)+str(topmost)+str(bottommost)