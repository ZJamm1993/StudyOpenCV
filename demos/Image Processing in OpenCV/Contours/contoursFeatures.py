import numpy as np 
from cv2 import cv2


img = cv2.imread('resource/hand.jpg')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

# moments
# Image moments help you to calculate some features like center of mass of the object, area of the object etc. Check out the wikipedia page on Image Moments

# The function cv2.moments() gives a dictionary of all moment values calculated. See below:

cnt = contours[0]
M = cv2.moments(cnt)
print M

# From this moments, you can extract useful data like area, centroid etc. Centroid is given by the relations

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

# Contour Area
# Contour area is given by the function cv2.contourArea() or from moments, M[‘m00’].
area = cv2.contourArea(cnt)

# Contour Perimeter
# It is also called arc length. It can be found out using cv2.arcLength() function. Second argument specify whether shape is a closed contour (if passed True), or just a curve.

perimeter = cv2.arcLength(cnt, True)

# Contour Approximation

# It approximates a contour shape to another shape with less number of vertices depending upon the precision we specify. It is an implementation of Douglas-Peucker algorithm. Check the wikipedia page for algorithm and demonstration.

# To understand this, suppose you are trying to find a square in an image, but due to some problems in the image, you didn’t get a perfect square, but a “bad shape” (As shown in first image below). Now you can use this function to approximate the shape. In this, second argument is called epsilon, which is maximum distance from contour to approximated contour. It is an accuracy parameter. A wise selection of epsilon is needed to get the correct output.
accs = [0.05, 0.01, 0.001]
for acc in accs:
    epsilon = acc * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # print approx
    approxcon = cv2.drawContours(img.copy(), [approx], -1, (0,255,0), 3)
    cv2.imshow('approx' + str(acc), approxcon)

# Convex Hull

# Convex Hull will look similar to contour approximation, but it is not (Both may provide same results in some cases). Here, cv2.convexHull() function checks a curve for convexity defects and corrects it. Generally speaking, convex curves are the curves which are always bulged out, or at-least flat. And if it is bulged inside, it is called convexity defects. For example, check the below image of hand. Red line shows the convex hull of hand. The double-sided arrow marks shows the convexity defects, which are the local maximum deviations of hull from contours.

# There is a little bit things to discuss about it its syntax:

# hull = cv2.convexHull(points[, hull[, clockwise[, returnPoints]]
# Arguments details:

# points are the contours we pass into.
# hull is the output, normally we avoid it.
# clockwise : Orientation flag. If it is True, the output convex hull is oriented clockwise. Otherwise, it is oriented counter-clockwise.
# returnPoints : By default, True. Then it returns the coordinates of the hull points. If False, it returns the indices of contour points corresponding to the hull points.
# So to get a convex hull as in above image, following is sufficient:
hull = cv2.convexHull(cnt)
hullcon = cv2.drawContours(img.copy(), [hull], -1, (0,0,255), 3)
cv2.imshow('hull', hullcon)

# Checking Convexity

# There is a function to check if a curve is convex or not, cv2.isContourConvex(). It just return whether True or False. Not a big deal.
print cv2.isContourConvex(cnt)

# Bounding Rectangle

# There are two types of bounding rectangles.

# a. Straight Bounding Rectangle

# It is a straight rectangle, it doesn’t consider the rotation of the object. So area of the bounding rectangle won’t be minimum. It is found by the function cv2.boundingRect().

# Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
x,y,w,h = cv2.boundingRect(cnt)
print cv2.boundingRect(cnt)
straightRect = cv2.rectangle(img.copy(),(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('straightRect', straightRect)

# b. Rotated Rectangle

# Here, bounding rectangle is drawn with minimum area, so it considers the rotation also. The function used is cv2.minAreaRect(). It returns a Box2D structure which contains following detals - ( top-left corner(x,y), (width, height), angle of rotation ). But to draw this rectangle, we need 4 corners of the rectangle. It is obtained by the function cv2.boxPoints()

minRect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(minRect)
box = np.int_(box) # need integer
rotatedRect = cv2.drawContours(img.copy(), [box], -1, (255,255,0),3)
cv2.imshow('rotatedRect',rotatedRect)

# Minimum Enclosing Circle

# Next we find the circumcircle of an object using the function cv2.minEnclosingCircle(). It is a circle which completely covers the object with minimum area.
(x,y),radius = cv2.minEnclosingCircle(cnt)
minEnclosCircle = cv2.circle(img.copy(), (int(x),int(y)), int(radius), (255,255,0), 2)
cv2.imshow('minEnclosCircle',minEnclosCircle)

# Fitting an Ellipse

# Next one is to fit an ellipse to an object. It returns the rotated rectangle in which the ellipse is inscribed.

ellipse = cv2.fitEllipse(cnt)
fitEllipse = cv2.ellipse(img.copy(),ellipse,(0,255,0),2)
cv2.imshow('fitEllipse',fitEllipse)

# Fitting a Line

# Similarly we can fit a line to a set of points. Below image contains a set of white points. We can approximate a straight line to it.
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
fitLine = cv2.line(img.copy(),(cols-1,righty),(0,lefty),(0,255,0),2)
cv2.imshow('fitLine',fitLine)

# hold window
cv2.waitKey(0)
cv2.destroyAllWindows()