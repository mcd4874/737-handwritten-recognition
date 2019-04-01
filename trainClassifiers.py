from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import matplotlib.pyplot as plt
import math
import xml.etree.ElementTree as ET
import os
import cv2
import pickle

def stackFeatures(stackFile, sampleListFile):
    # Load the dictionary file
    data = np.genfromtxt(sampleListFile, delimiter=',', dtype=str)

    # Load the stackFile
    stackCache = np.genfromtxt(stackFile, delimiter=',', dtype=float)

    stack = None

    # Iterate over the lines in the sampleList file
    sampleSize = len(data)
    # sampleSize = 100
    # for i in range(len(data)):
    targetClasses = list()
    for i in range(sampleSize):
        # Extract the target class
        target = data[i][1].strip()
        targetClasses.append(target)
        # Extract the unique identifier for the symbol
        elements = str(data[i][0]).split("_")
        id = elements[len(elements) - 1]
        #filename = imagesPath + id + ".png"
        # print("filename=",  filename)
        #img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        #print("img =", img)
        #print("img type =", type(img))

        #imgRavel = img.ravel()
        # print(imgRavel.shape)
        #print("imgRavel = ", imgRavel, " shape is ", imgRavel.shape)

        # Lazily allocate the stack array
        if stack is None:
            #height = img.shape[0]
            #width = img.shape[1]
            #print("height and width = ", height, " and ", width)
            #print("length = ", len(data))
            stack = np.zeros((sampleSize, stackCache.shape[1]))

        # Merge this flattened image into our stack
        stack[i] =  stackCache[int(id)]
        print("i=", i, ", id=", id)
    targetClasses = np.array(targetClasses, dtype=np.dtype('a16'))
    # print(targetClasses.shape)
    # print(stack.shape)
    return stack,targetClasses

def trainKDTreeClassifier(stack,targetClasses):
    # print("About to run KDTree")
    # kdtree = KDTree(stack)
    # print("About to query KDTree")
    # dist, ind = kdtree.query(stack[1], k=3)
    # print("indices of 3 closest = ", ind)
    # print("distances of 3 closest = ", dist)

    kdTree = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree')

    # Train with the data skipping the 0th column containing the UI
    kdTree.fit(stack[:, 1:],targetClasses)

    return kdTree

def trainRandomForestClassifier(stack, targetClasses, maxTrees, maxDepth):
    rf = RandomForestClassifier(n_estimators=maxTrees, max_depth=maxDepth)

    # Train with the data skipping the 0th column containing the UI
    rf.fit(stack[:, 1:],targetClasses)
    return rf

def testKDTreeClassifier(testSamplesFile, labelTestTarget, kdtree, encoderModel):
    predict = kdtree.predict(testSamplesFile[:, 1:])
    print("prediction: ",predict)
    print("actual label : ",labelTestTarget)
    #print(confusion_matrix(labelTestTarget, predict, labels=encoderModel.classes_))
    print(confusion_matrix(labelTestTarget, predict, labels=None))
    #print(classification_report(labelTestTarget, predict, target_names=encoderModel.classes_))
    print(classification_report(labelTestTarget, predict, target_names=None))
    return

def testRandomForestClassifier(testSamplesFile, labelTestTarget, rf, encoderModel):
    predict = rf.predict(testSamplesFile[:, 1:])
    print("rf prediction: ", predict)
    print("rf actual label: ", labelTestTarget)
    print(confusion_matrix(labelTestTarget, predict, labels=None))
    print(classification_report(labelTestTarget, predict, target_names=None))


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
    maxTrees = 100
    maxDepth = 20

    #trainSymbols,targetSymbols = stackFeatures("./symbolStack.csv", "./trainingSymbols/iso_GT_train_resampled.txt")
    #testSymbols,testTargetSymbols = stackFeatures("./junkStack.csv", "./trainingSymbols/iso_GT_test_resampled.txt")

    # Don't use resampled dataset because this is simulating KNN-1; resampled data is not adding any value
    trainSymbols, trainTargetSymbols = stackFeatures("./symbolStack.csv", "./trainingSymbols/iso_GT_train.txt")
    testSymbols, testTargetSymbols = stackFeatures("./symbolStack.csv", "./trainingSymbols/iso_GT_test.txt")
    #trainSymbols, trainTargetSymbols = stackFeatures("./symbolStack.csv", "./trainingSymbols/iso_GT_train_resampled.txt")
    #testSymbols, testTargetSymbols = stackFeatures("./symbolStack.csv", "./trainingSymbols/iso_GT_test_resampled.txt")

    encoderModel = generateLabelsEncoder(trainTargetSymbols)
    # encoderModel = generateLabelsEncoder(testTargetSymbols)

    print(encoderModel.classes_)
    labelTrainTarget = transformLabels(trainTargetSymbols,encoderModel)
    labelTestTarget = transformLabels(testTargetSymbols,encoderModel)
    print("train target: ",labelTrainTarget)
    print("test target: ",labelTestTarget)

    #kdtree = trainKDTreeClassifier(trainSymbols, trainTargetSymbols)
    #testKDTreeClassifier(testSymbols,testTargetSymbols, kdtree,encoderModel)

    rf = trainRandomForestClassifier(trainSymbols, trainTargetSymbols, maxTrees, maxDepth)

    # Save the RandomForestClassifier for later use
    # https://stackabuse.com/scikit-learn-save-and-restore-models/


    testRandomForestClassifier(testSymbols, testTargetSymbols, rf, encoderModel)
    return

main()