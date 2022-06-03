import cv2
import numpy as np
import copy
import time

epsilon = 0.1

def imgResize(img,rate=0.5):
    return cv2.resize(img, dsize=(0, 0), fx=rate, fy=rate, interpolation=cv2.INTER_LINEAR)

def eucliDist(pt1,pt2):
    return np.sqrt(np.sum(np.square(pt1-pt2)))

def centGen(img,num):
    centroids = np.random.randint(img[0,:,:].reshape(-1).shape,size=num)
    centroids = np.array((centroids//img[0,0,:].shape,centroids%img[0,0,:].shape)).T
    return centroids

def kMeans(img,k=5):
    '''
    '''
    if len(img.shape)==3:
        img = np.array([img[:,:,0],img[:,:,1],img[:,:,2]])
    elif len(img.shape)==1:
        img = img.reshape((1,)+img.shape)
    pixelData = img
    centroids = centGen(img,k)
    height,width = img[0,:,:].shape
    centCurrent = [0 for i in range(k)]
    for i in range(k):
        centCurrent[i] = pixelData[:,centroids[i][0],centroids[i][1]]
    print("current centroid : \n",centCurrent)
    condition = 1
    while condition==True:
        kIndex = []
        count = [0 for i in range(k)]
        centUpdate = np.zeros(shape=[k,3],dtype=np.float32)
        for y in range(height):
            for x in range(width):
                dist = []
                for i in range(k):
                    dist.append(eucliDist(centCurrent[i],pixelData[:,y,x]))
                    # print(dist)
                kIndex.append([np.argmin(dist)]) # max index
        for i in range(k):
            for l in range(img[0,:,:].reshape(-1).shape[0]):
                if kIndex[l][0]==i:
                    centUpdate[i]+=pixelData[:,l//img[0,0,:].shape[0],l%img[0,0,:].shape[0]]
                    count[i]+=1
        centUpdate = np.array(centUpdate)
        for i in range(k):
            if count[i]!=0:
                centUpdate[i] = centUpdate[i]/count[i]
            else:
                centUpdate[i] = centCurrent[i]
        print("update centroid : \n",centUpdate)
        print("count : ", count)
        if np.sum(np.abs(np.array(centUpdate)-np.array(centCurrent)))<epsilon:
            condition = 0
        centCurrent = centUpdate
    return centUpdate, kIndex

def kMeansVis(img,centroids,kIndex):
    '''
    '''
    if len(img.shape)==3:
        imgShape = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape)==2:
        imgShape = img
    height,width = imgShape.shape
    kMeansImg = np.zeros(shape=img.shape,dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            kMeansImg[y,x] = centroids[kIndex[img.shape[1]*y+x],:3]
    return kMeansImg

def main():
    img0 = cv2.imread('/home/hyh/cat.jpg', cv2.IMREAD_COLOR)
    img0 = imgResize(img0,0.05)

    centroids,kIndex = kMeans(img0,k=3)
    kMeansImg = kMeansVis(img0,centroids,kIndex)

    cv2.imshow('original image',img0)
    cv2.imshow('k means clustering RGB image',kMeansImg)
    cv2.waitKey(0)

if __name__=='__main__':
    main()
