# try to implemente the K-Means

import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

def distance(obj0, obj1):
    size = obj0.shape[0]
    tota = 0
    for i in range(size):
        tota += np.square(obj0[i] - obj1[i])
    res = np.sqrt(tota)
    return res

def average(objs):
    count = objs.shape[0]
    avgsize = objs.shape[1]
    avg = np.zeros((avgsize), dtype = np.float32)
    for obj in objs:
        for i in range(avgsize):
            avg[i] += obj[i]
    avg = avg / count
    return avg

# returns ret, labels, centers
def myKMeans(samples, kCount, maxIter = 50):
    sampleCount = samples.shape[0]
    labels = np.zeros((sampleCount), dtype = np.int)
    max = samples.max()
    min = samples.min()
    centers = np.float32(np.random.randint(min, max, (kCount, samples.shape[1])))

    for _ in range(maxIter):
        # break
        # step 1: find each sample's nearest center
        for samindex in range(sampleCount):
            sam = samples[samindex]
            testdis = 10000000000
            for cenindex in range(kCount):
                cent = centers[cenindex]
                dis = distance(sam, cent)
                if dis <= testdis:
                    testdis = dis
                    labels[samindex] = cenindex
        
        # step 2: find new centers with samples' avgs
        for i in range(kCount):
            clus = samples[labels == i]
            newcent = average(clus)
            if np.isnan(newcent[0]):
                newcent = average(samples)
            centers[i] = newcent
            
    return (True, labels, centers)

def testKMean():
    wei = 0
    count = 5

    Z = np.random.randint(25, 50, (1, 2))
    for i in range(count):
        rand = np.random.randint(wei, wei + 200, (50, 2))
        Z = np.vstack((Z, rand))
        wei = wei + 150

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria and apply kmeans()
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # _, label, center = cv2.kmeans(Z, count, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, label, center = myKMeans(Z, count)
    print(center)

    # Now separate the data, Note the flatten()
    colors = ['red', 'green', 'm', 'c', 'y', 'r', 'g', 'm', 'c', 'y', 'r', 'g', 'm', 'c', 'y']
    ravel = label.ravel()
    for i in range(count):
        A = Z[ravel == i]
        plt.scatter(A[:, 0], A[:, 1],c = colors[i])
    # Plot the data
    plt.scatter(center[:, 0], center[:, 1], s = 100, marker = 's', c = 'orange')
    plt.xlabel('Height'), plt.ylabel('Weight')
    plt.show()

if __name__ == "__main__":
    # a = np.array([0, 0, 0, 0], dtype = np.float)
    # b = np.array([400, 300, 500, 600], dtype = np.float)
    # c = np.array([400, 300, 500, 600], dtype = np.float)
    # dis = distance(a, b)
    # objs = np.array((a, b, c))
    # avg = average(objs)
    # print(avg)
    # print(dis)

    testKMean()