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

    # Create a dictionary from UI to stackCache index

    stack = None

    # Iterate over the lines in the sampleList file
    sampleSize = len(data)
    targetClasses = list()
    for i in range(sampleSize):
        # Extract the target class
        target = data[i][1].strip()
        targetClasses.append(target)
        # Extract the unique identifier for the symbol
        elements = str(data[i][0]).split("_")
        id = elements[len(elements) - 1]

        # Lazily allocate the stack array
        if stack is None:
            stack = np.zeros((sampleSize, stackCache.shape[1]))

        # Merge this flattened image into our stack
        stack[i] = stackCache[int(id)]
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


def trainKDTreeClassifier(stack, targetClasses):
    kdTree = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree')
    # Train with the data skipping the 0th column containing the UI
    kdTree.fit(stack[:, 1:], targetClasses)

    return kdTree


def trainRandomForestClassifier(stack, targetClasses, maxTrees, maxDepth):
    rf = RandomForestClassifier(n_estimators=maxTrees, max_depth=maxDepth)
    # Train with the data skipping the 0th column containing the UI
    rf.fit(stack[:, 1:], targetClasses)
    return rf


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
    kdtree= KNeighborsClassifier(n_neighbors=1,algorithm = 'kd_tree' )
    kdtree.fit(trainSymbols[:, 1:], labelTrainTarget)
    with open(modelPath, 'wb') as file:
       pickle.dump(kdtree, file, -1)
    print("finish train kd tree")
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
    rf = RandomForestClassifier(n_estimators=maxTrees, max_depth=maxDepth)
    rf = rf.fit(trainSymbols[:, 1:], labelTrainTarget)

    # Save the RandomForestClassifier for later use
    # https://stackabuse.com/scikit-learn-save-and-restore-models/
    # pkl_filename = "pickle_rf.pkl"
    with open(modelPath, 'wb') as file:
        pickle.dump(rf, file, -1)
    print("finish train random forest")
    return

def main():

    # Don't use resampled dataset because this is simulating KNN-1; resampled data is not adding any value
    trainSymbols, trainTargetSymbols = stackFeatures("./symbolStack.csv", "./trainingSymbols/iso_GT_train.txt")

    trainTrunk, trainTargetTrunk = stackFeatures("./junkStack.csv", "./trainingJunk/junk_GT_train.txt")


    trainJunkSymbols = np.concatenate((trainSymbols, trainTrunk), axis=0)
    targetJunkSymbols = np.concatenate((trainTargetSymbols, trainTargetTrunk), axis=0)

    print(trainSymbols.shape)
    print(trainJunkSymbols.shape)

    print(np.isnan(trainSymbols[:, 1:]).any())
    train_rf_model(trainSymbols, trainTargetSymbols, "encoder_rf.pkl", "pickle_rf.pkl")
    train_kd_model(trainSymbols, trainTargetSymbols, "encoder_kD.pkl", "pickle_kd.pkl")

    # train both junk and symbol

    train_rf_model(trainJunkSymbols,targetJunkSymbols,"encoder_both_rf.pkl","pickle_both_rf.pkl")
    train_kd_model(trainJunkSymbols,targetJunkSymbols,"encoder_both_kD.pkl","pickle_both_kd.pkl")
    return

main()