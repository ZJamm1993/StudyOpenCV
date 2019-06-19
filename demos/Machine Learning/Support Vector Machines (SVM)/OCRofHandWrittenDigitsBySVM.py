from cv2 import cv2
import numpy as np

SZ=20
bin_n = 16 # Number of bins

affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    # cv2.imshow('hog',hist)
    return hist

img = cv2.imread('resource/digits.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

# First half is trainData, remaining is testData
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]

######     Now training      ########################

deskewed = []
hogdata = []
for row in train_cells:
    for i in row:
        deske = deskew(i)
        hogg = hog(deske)
        deskewed.append(deske)
        hogdata.append(hogg)
        # cv2.imshow('deskew',deske)
        # cv2.imshow('hog',hogg)
        # cv2.waitKey(5)
# deskewed = [map(deskew,row) for row in train_cells]
# hogdata = [map(hog,row) for row in deskewed]
trainData = np.array(hogdata).reshape(-1,64).astype(np.float32)
responses = np.repeat(np.arange(10),250)[:,np.newaxis]
# responses = responses.astype(np.float32)

svm = cv2.ml.SVM_create()
# svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR,
#                     svm_type = cv2.ml.SVM_C_SVC,
#                     C=2.67, gamma=5.383 )
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(trainData,0,responses)
svm.save('svm_data.dat')

######     Now testing      ########################

deskewed = [map(deskew,row) for row in test_cells]
hogdata = [map(hog,row) for row in deskewed]
testData = np.array(hogdata).reshape(-1,bin_n*4).astype(np.float32)
result = np.array(svm.predict(testData))
result = result[1]

#######   Check Accuracy   ########################
mask = result==responses
correct = np.count_nonzero(mask)
print correct*100.0/result.size

# 91.56 while no deskewing
# 93.8 while deskewing
