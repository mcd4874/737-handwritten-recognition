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
import pandas as pd
import csv

def stackFeatures(stackFile, sampleListFile):
    # Load the dictionary file
    data = np.genfromtxt(sampleListFile, delimiter=',', dtype=str)

    # Load the stackFile
    stackCache = np.genfromtxt(stackFile, delimiter=',', dtype=float)

    stack = None
    uiStack = []

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
        ui = data[i][0];
        uiStack.append(ui.strip())
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
        # print("i=", i, ", id=", id)
    targetClasses = np.array(targetClasses, dtype=np.dtype('a16'))
    # print(targetClasses.shape)
    # print(stack.shape)
    return stack, targetClasses, uiStack

def testKDTreeClassifier(testSamplesFile, labelTestTarget, kdtree, encoderModel, uiStack):
    predict = kdtree.predict(testSamplesFile[:, 1:])
    #dist, ind = kdtree.query(testSamplesFile[:, 1:])
    #print("ind = ", ind)

    print("prediction: ",predict)
    print("actual label : ",labelTestTarget)
    print(confusion_matrix(labelTestTarget, predict, labels=None))
    print(classification_report(labelTestTarget, predict, target_names=None))
    return

def cleanString(string):
    elements = str(string).split("'")
    return elements[1];

def testRandomForestClassifier(testSamplesFile, labelTestTarget, rf, encoderModel, uiStack):
    predict = rf.predict(testSamplesFile[:, 1:])
    print("rf prediction: ", predict)
    print("rf actual label: ", labelTestTarget)
    print("Manual confusion matrix:")

    matrix = confusion_matrix(labelTestTarget, predict, labels=None)

    # Print Column Headers for confusion matrix
    print ("", end=",")  # Initial spacer for header row
    for i in range(len(matrix[0])):
        print (encoderModel.classes_[i], end=",")
    print("")
    for i in range(len(matrix)):
        # Prubt Row Headers for confusion matrix
        print (encoderModel.classes_[i], end=",")
        for j in range(len(matrix[i])):
            print (matrix[i][j], end=",")
        print("")
    print(confusion_matrix(labelTestTarget, predict, labels=None))
    print(classification_report(labelTestTarget, predict, target_names=encoderModel.classes_))
    return


def transformLabels(targetClasses,encoderModel):
    return encoderModel.transform(targetClasses)



def get_list_indices_predict(testSamplesFile, classifier, encoderModel):
    """

    :param testSamplesFile:
    :param labelTestTarget:
    :param classifier:
    :param encoderModel:
    :return:
    """
    resultList = []
    predict_probs = classifier.predict_proba(testSamplesFile[:, 1:])
    # print("sahpe of probs : ", predict_probs.shape)
    sort_predict_probs_indices = predict_probs.argsort(axis = 1)[:,-10:]
    result = np.chararray(sort_predict_probs_indices.shape, itemsize=20)
    for i in range(result.shape[0]):
        resultVector = []
        for j in range(result.shape[1]):
            result[i][j] = encoderModel.classes_[sort_predict_probs_indices[i][j]]
            resultVector.append(cleanString(result[i][j]))
        result[i] = result[i][::-1]
        resultList.append(resultVector)
    return resultList

def test_model(testSymbols,labelTestTarget, stack, encoderPath, modelPath,reportPath):
    """

    :param testSymbols:
    :param labelTestTarget:
    :param stack:
    :param encoderPath:
    :param modelPath:
    :param reportPath:
    :return:
    """
    with open(encoderPath, 'rb') as file:
        encoderModel = pickle.load(file)
    print("Finished loading encoder.")

    labelTestTarget = transformLabels(labelTestTarget, encoderModel)
    print("Loading random forest from pickle...")
    with open(modelPath, 'rb') as file:
        rf = pickle.load(file)
    print("Finished loading.")

    testRandomForestClassifier(testSymbols, labelTestTarget, rf, encoderModel, stack)
    list_test_best_label_predict = get_list_indices_predict(testSymbols, rf, encoderModel)
    df1 = pd.DataFrame(stack)
    df2 = pd.DataFrame(list_test_best_label_predict)
    r = pd.concat([df1, df2], axis=1)
    r.to_csv(reportPath, index=False, header=False, encoding="utf-8", quoting=csv.QUOTE_NONE)


def main():
    testSymbols, testTargetSymbols, uiStack = stackFeatures("./symbolStack.csv", "./trainingSymbols/iso_GT_test.txt")
    testJunks, testTargetJunk, junkStack = stackFeatures("./junkStack.csv", "./trainingJunk/junk_GT_test.txt")

    testJunkSymbols = np.concatenate((testSymbols, testJunks), axis=0)
    testTargetJunkSymbols = np.concatenate((testTargetSymbols, testTargetJunk), axis=0)
    uiJunkStack = np.concatenate((uiStack, junkStack), axis=0)

    encoderPath = "encoder_rf.pkl"
    rfModelPath = "pickle_rf.pkl"
    report_table_rf = "report_table_rf.csv"
    kdModelPath = "pickle_kd.pkl"
    report_table_kd = "report_table_kd.csv"
    test_model(testSymbols, testTargetSymbols, uiStack, encoderPath, rfModelPath,report_table_rf)
    test_model(testSymbols, testTargetSymbols, uiStack, encoderPath, kdModelPath,report_table_kd)

    encoderBothPath = "encoder_both_rf.pkl"
    rfModelBothPath = "pickle_both_rf.pkl"
    report_table_both_rf = "report_table_both_rf.csv"
    kdModelBothPath = "pickle_both_kd.pkl"
    report_table_both_kd = "report_table_both_kd.csv"

    test_model(testJunkSymbols, testTargetJunkSymbols, uiJunkStack, encoderBothPath, rfModelBothPath,report_table_both_rf)
    test_model(testJunkSymbols, testTargetJunkSymbols, uiJunkStack, encoderBothPath, kdModelBothPath,report_table_both_kd)

main()