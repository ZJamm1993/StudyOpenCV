# it's going wrong when add or sub if their sizes are not same 
# it may change size if pyrDown() then pyrUp(), so resize it 

from cv2 import cv2
import numpy as np,sys

A = cv2.imread('resource/apple.jpg')
B = cv2.imread('resource/orange.jpg')

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in xrange(6):
    G = cv2.pyrDown(G)
    # cv2.imshow('gpa'+ str(i), G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in xrange(6):
    G = cv2.pyrDown(G)
    gpB.append(G)


# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in xrange(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    GE_1 = cv2.resize(gpA[i-1], GE.shape[0:2])
    # cv2.imshow('ge'+ str(i), GE_1)
    L = cv2.subtract(GE_1,GE)
    # cv2.imshow('lpa'+ str(i), L)
    lpA.append(L)


# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in xrange(5,0,-1):
    GE = cv2.pyrUp(gpB[i])
    GE_1 = cv2.resize(gpB[i-1], GE.shape[0:2])
    L = cv2.subtract(GE_1,GE)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
ii = 0
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
    # cv2.imshow('ls' + str(ii), ls)
    # ii = ii + 1
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in xrange(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.resize(ls_, LS[i].shape[0:2])
    ls_ = cv2.add(ls_, LS[i])

# image with direct connecting each half
real = np.hstack((A[:,:cols/2],B[:,cols/2:]))

cv2.imshow('apple', A)
cv2.imshow('orange', B)
cv2.imshow('blend direct', real)
cv2.imshow('pyramids blend', ls_)
# '''

# hold window
cv2.waitKey(0)
cv2.destroyAllWindows()