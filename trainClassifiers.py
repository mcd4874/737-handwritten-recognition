"""
Author: William Duong, Eric Hartman
This file will extract image data from csv file to train KD model and random forest model
"""


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
    """
    This function will extract image data from the stackFile and combine those data into
    an numpy array dataset. It will do the same for the target label
    :param stackFile: input csv file which contains image data
    :param sampleListFile: input txt file which contains identify to the csv file and the label
    :return: the stack train input and the stack target classes input
    """
    # Load the dictionary file
    data = np.genfromtxt(sampleListFile, delimiter=',', dtype=str)

    # Load the stackFile
    stackCache = np.genfromtxt(stackFile, delimiter=',', dtype=float)
    #stackCacheString = np.genfromtxt(stackFile, delimiter=',', dtype=str)

    # Create a dictionary from UI to stackCache index
    #uiToIndex = dict()
    #for i in range(len(stackCache)):
    #    uiToIndex[stackCacheString[i][0]] = i

    stack = None

    # Iterate over the lines in the sampleList file
    sampleSize = len(data)
    targetClasses = list()
    for i in range(sampleSize):
        # Extract the target class
        target = data[i][1].strip()
        targetClasses.append(target)
        # Extract the unique identifier for the symbol
        #ui = data[i][0]
        elements = str(data[i][0]).split("_")
        id = elements[len(elements) - 1]

        # Lazily allocate the stack array
        if stack is None:
            stack = np.zeros((sampleSize, stackCache.shape[1]))

        # Merge this flattened image into our stack
        #stack[i] =  stackCache[int(id)]
        #print("Looking up ui=", ui)
        #stack[i] = stackCache[uiToIndex[ui]]
        stack[i] = stackCache[int(id)]
        print("i=", i, ", id=", id)
    targetClasses = np.array(targetClasses, dtype=np.dtype('a16'))
    return stack,targetClasses


def generateLabelsEncoder(targetClasses):
    """
    create label encoder model with target classes label
    :param targetClasses: vector of target classes label
    :return: the encoder model
    """
    enc = LabelEncoder()
    enc.fit(targetClasses)
    return enc

def transformLabels(targetClasses,encoderModel):
    """
    This function will convert target classes labels into numeric vectors for one hot encoding
    :param targetClasses: vector of target classes label
    :param encoderModel: label encoder model contains target classes labels
    :return:vector of numeric label
    """
    return encoderModel.transform(targetClasses)


def train_kd_model(trainSymbols,trainTargetSymbols,encoderPath,modelPath):
    """
    this function will train a KNN model with n=1 using kd tree algorithm
    :param trainSymbols:input images data matrix
    :param trainTargetSymbols: input target classes vector
    :param encoderPath: file path to store the encoder model
    :param modelPath: file path to store the classifier model
    :return: model after train
    """
    encoderModel = generateLabelsEncoder(trainTargetSymbols)

    with open(encoderPath, 'wb') as file:
        pickle.dump(encoderModel, file, -1)

    labelTrainTarget = transformLabels(trainTargetSymbols, encoderModel)

    kdtree = trainKDTreeClassifier(trainSymbols, trainTargetSymbols)
    with open(modelPath, 'wb') as file:
       pickle.dump(kdtree, file, -1)
    return

def train_rf_model(trainSymbols,trainTargetSymbols,encoderPath,modelPath):
    """
    this function will train a random forest model
    :param trainSymbols:input images data matrix
    :param trainTargetSymbols: input target classes vector
    :param encoderPath: file path to store the encoder model
    :param modelPath: file path to store the classifier model
    :return: model after train
    """
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

    # Don't use resampled dataset because this is simulating KNN-1; resampled data is not adding any value
    trainSymbols, trainTargetSymbols = stackFeatures("./symbolStack.csv", "./trainingSymbols/iso_GT_train.txt")

    trainTrunk, trainTargetTrunk = stackFeatures("./junkStack.csv", "./trainingJunk/junk_GT_train.txt")


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