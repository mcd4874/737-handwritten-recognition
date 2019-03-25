from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import math
import xml.etree.ElementTree as ET
import os
import cv2

def stackImages(imagesPath, sampleListFile):
    # Load the dictionary file
    data = np.genfromtxt(sampleListFile, delimiter=',', dtype=str)
    stack = None

    # Iterate over the lines in the sampleList file
    # sampleSize = len(data)
    sampleSize = 10
    # for i in range(len(data)):
    targetClasses = list()
    for i in range(sampleSize):
        # Extract the target class
        target = data[i][1]
        targetClasses.append(target)
        # Extract the unique identifier for the symbol
        elements = str(data[i][0]).split("_")
        id = elements[len(elements) - 1]
        filename = imagesPath + id + ".png"
        print("filename=",  filename)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        #print("img =", img)
        #print("img type =", type(img))

        imgRavel = img.ravel()
        print(imgRavel.shape)
        #print("imgRavel = ", imgRavel, " shape is ", imgRavel.shape)

        # Lazily allocate the stack array
        if stack is None:
            height = img.shape[0]
            width = img.shape[1]
            #print("height and width = ", height, " and ", width)
            #print("length = ", len(data))
            stack = np.zeros((sampleSize, height*width))

        # Merge this flattened image into our stack
        stack[i] = imgRavel
    targetClasses = np.array(targetClasses, dtype=np.dtype('a16'))
    # print(targetClasses.shape)
    # print(stack.shape)
    return stack,targetClasses

def trainKDTreeClassifier(stack,targetClasses):
    print("About to run KDTree")
    kdtree = KDTree(stack)
    print("About to query KDTree")
    dist, ind = kdtree.query(stack[1], k=3)
    print("indices of 3 closest = ", ind)
    print("distances of 3 closest = ", dist)

    return

def testKDTreeClassifier(testSamplesFile, kdtree):
    return

def generateLabelsEncoder(targetClasses):
    enc = LabelEncoder()
    enc.fit(targetClasses)
    return enc

def transformLabels(targetClasses,encoderModel):
    return encoderModel.transform(targetClasses)

def inverseTransformLabels(targetClasses,encoderModel):
    return encoderModel.inverse_transform(targetClasses)

def main():
    trainSymbols,targetSymbols = stackImages("./images/connect/symbols/", "./trainingSymbols/iso_GT_train_resampled.txt")
    #testSymbols = stackImages("./images/connect/junk/", "./trainingSymbols/iso_GT_test_resampled.txt")

    # kdtree = trainKDTreeClassifier(trainSymbols,targetSymbols)
    #testKDTreeClassifier(testSymbols, kdtree)

    return

main()