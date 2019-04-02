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
    kdTree = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree')
    # Train with the data skipping the 0th column containing the UI
    kdTree.fit(stack[:, 1:],targetClasses)

    return kdTree

def trainRandomForestClassifier(stack, targetClasses, maxTrees, maxDepth):
    rf = RandomForestClassifier(n_estimators=maxTrees, max_depth=maxDepth)
    # Train with the data skipping the 0th column containing the UI
    rf.fit(stack[:, 1:],targetClasses)
    return rf

def generateLabelsEncoder(targetClasses):
    enc = LabelEncoder()
    enc.fit(targetClasses)
    return enc

def transformLabels(targetClasses,encoderModel):
    return encoderModel.transform(targetClasses)

def inverseTransformLabels(targetClasses,encoderModel):
    return encoderModel.inverse_transform(targetClasses)

<<<<<<< HEAD


def train_kd_model(trainSymbols,trainTargetSymbols,encoderPath,modelPath):
    encoderModel = generateLabelsEncoder(trainTargetSymbols)

    with open(encoderPath, 'wb') as file:
        pickle.dump(encoderModel, file, -1)

    labelTrainTarget = transformLabels(trainTargetSymbols, encoderModel)

    kdtree = trainKDTreeClassifier(trainSymbols, trainTargetSymbols)
    with open(modelPath, 'wb') as file:
       pickle.dump(kdtree, file, -1)
    return

def train_rf_model(trainSymbols,trainTargetSymbols,encoderPath,modelPath):
    encoderModel = generateLabelsEncoder(trainTargetSymbols)

    with open(encoderPath, 'wb') as file:
        pickle.dump(encoderModel, file, -1)

    labelTrainTarget = transformLabels(trainTargetSymbols, encoderModel)

    maxTrees = 100
    maxDepth = 20
    rf = trainRandomForestClassifier(trainSymbols, labelTrainTarget, maxTrees, maxDepth)

    # Save the RandomForestClassifier for later use
    # https://stackabuse.com/scikit-learn-save-and-restore-models/
    # pkl_filename = "pickle_rf.pkl"
    with open(modelPath, 'wb') as file:
        pickle.dump(rf, file, -1)
    return

def main():


    trainTrunk, trainTargetTrunk = stackFeatures("./junkStack.csv", "./trainingJunk/junk_GT_train.txt")

    # Don't use resampled dataset because this is simulating KNN-1; resampled data is not adding any value
    trainSymbols, trainTargetSymbols = stackFeatures("./symbolStack.csv", "./trainingSymbols/iso_GT_train.txt")

    trainJunkSymbols = np.concatenate((trainSymbols, trainTrunk), axis=0)
    targetJunkSymbols = np.concatenate((trainTargetSymbols, trainTargetTrunk), axis=0)

    print(trainSymbols.shape)
    print(trainJunkSymbols.shape)

    train_rf_model(trainSymbols, trainTargetSymbols, "encoder_rf.pkl", "pickle_rf.pkl")
    train_kd_model(trainSymbols, trainTargetSymbols, "encoder_kD.pkl", "pickle_kd.pkl")

    # train both junk and symbol
    
    train_rf_model(trainJunkSymbols,targetJunkSymbols,"encoder_both_rf.pkl","pickle_both_rf.pkl")
    train_kd_model(trainJunkSymbols,targetJunkSymbols,"encoder_both_kD.pkl","pickle_both_kd.pkl")







    return

main()