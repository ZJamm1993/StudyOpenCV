# OCR of Hand-written Digits

# Our goal is to build an application which can read the handwritten digits. For this we need some train_data and test_data. OpenCV comes with an image digits.png (in the folder opencv/samples/python2/data/) which has 5000 handwritten digits (500 for each digit). Each digit is a 20x20 image. So our first step is to split this image into 5000 different digits. For each digit, we flatten it into a single row with 400 pixels. That is our feature set, ie intensity values of all pixels. It is the simplest feature set we can create. We use first 250 samples of each digit as train_data, and next 250 samples as test_data. So let’s prepare them first.

import numpy as np 
from cv2 import cv2

img = cv2.imread('resource/digits.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 50 x 100 cells, 20 x 20 each
rows = np.vsplit(gray, 50)
cells = [np.hsplit(row, 100) for row in rows]

x = np.array(cells)

# Now we prepare train_data and test_data.
# train is the left half, and the test is the right half. then reshape to (2500,400)
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400) 
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
knn.train(train,0,train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=5)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print accuracy

print 'hello'

# So our basic OCR app is ready. This particular example gave me an accuracy of 91%. One option improve accuracy is to add more data for training, especially the wrong ones. So instead of finding this training data everytime I start application, I better save it, so that next time, I directly read this data from a file and start classification. You can do it with the help of some Numpy functions like np.savetxt, np.savez, np.load etc. Please check their docs for more details.

# # save the data
# np.savez('knn_data.npz',train=train, train_labels=train_labels)

# # Now load the data
# with np.load('knn_data.npz') as data:
#     print data.files
#     train = data['train']
#     train_labels = data['train_labels']
# In my system, it takes around 4.4 MB of memory. Since we are using intensity values (uint8 data) as features, it would be better to convert the data to np.uint8 first and then save it. It takes only 1.1 MB in this case. Then while loading, you can convert back into float32.