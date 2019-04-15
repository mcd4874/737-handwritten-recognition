from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from generateFeatureStack import getFeatures

import numpy as np
import matplotlib.pyplot as plt
import math
import xml.etree.ElementTree as ET
import os
import cv2
import pickle
import pandas as pd
import csv
import sys


def cleanString(string):
    elements = str(string).split("'")
    return elements[1];

def testClassifier(testSamplesFile, labelTestTarget, rf, encoderModel, uiStack):
    predict = rf.predict(testSamplesFile[:, 0:])
    combine = np.concatenate((predict, labelTestTarget), axis=0)
    uniquesLabelSets = np.unique(combine)
    labelTags = list()
    for i in range(len(uniquesLabelSets)):
        labelTags.append(encoderModel.classes_[uniquesLabelSets[i]])
    print("tag target: ", labelTags)

    # only display recognition rates if labelTestTarget is provided
    if len(labelTestTarget) > 0:
        print("rf prediction: ", predict)
        print("rf actual label: ", labelTestTarget)
        print("Manual confusion matrix:")

        matrix = confusion_matrix(labelTestTarget, predict, labels=None)

        # Print Column Headers for confusion matrix
        print ("", end=",")  # Initial spacer for header row
        for i in range(len(matrix[0])):
            print (labelTags[i], end=",")
        print("")
        for i in range(len(matrix)):
            # Prubt Row Headers for confusion matrix
            print (labelTags[i], end=",")
            for j in range(len(matrix[i])):
                print (matrix[i][j], end=",")
            print("")
        print(confusion_matrix(labelTestTarget, predict, labels=None))
        print(classification_report(labelTestTarget, predict, target_names=labelTags))
    return


def transformLabels(targetClasses,encoderModel):
    """
    This function will convert target classes labels into numeric vectors for one hot encoding
    :param targetClasses: vector of target classes label
    :param encoderModel: label encoder model contains target classes labels
    :return:vector of numeric label
    """
    return encoderModel.transform(targetClasses)


def get_list_indices_predict(testSamplesFile, classifier, encoderModel):
    resultList = []

    predict_probs = classifier.predict_proba(testSamplesFile[:, 0:])

    sort_predict_probs_indices = predict_probs.argsort(axis = 1)[:,-10:]

    result = np.chararray(sort_predict_probs_indices.shape, itemsize=20)
    for i in range(result.shape[0]):
        resultVector = []
        for j in range(result.shape[1]):
            result[i][j] = encoderModel.classes_[sort_predict_probs_indices[i][j]]
            resultVector.append(cleanString(result[i][j]))
        result[i] = result[i][::-1]

        # Reverese the results so it is in decreasing accuracy order...
        resultVector.reverse()
        resultList.append(resultVector)
    return resultList

def test_model(testSymbols, labelTestTarget, stack, encoderPath, modelPath, outputFile):
    with open(encoderPath, 'rb') as file:
        encoderModel = pickle.load(file)
    print("Finished loading encoder.")
    print(encoderModel.classes_)

    labelTestTarget = transformLabels(labelTestTarget, encoderModel)


    # Load the kdtree from our pickle

    # Load the RandomForestClassifier from our pickle
    # https://stackabuse.com/scikit-learn-save-and-restore-models/
    print("Loading model from pickle...")
    with open(modelPath, 'rb') as file:
        rf = pickle.load(file)
    print("Finished loading.")

    testClassifier(testSymbols, labelTestTarget, rf, encoderModel, stack)
    list_test_best_label_predict = get_list_indices_predict(testSymbols, rf, encoderModel)
    df1 = pd.DataFrame(stack)
    df2 = pd.DataFrame(list_test_best_label_predict)
    r = pd.concat([df1, df2], axis=1)
    r.to_csv(outputFile, index=False, header=False, encoding="utf-8", quoting=csv.QUOTE_NONE)


def main():
    # Print usage...
    if(len(sys.argv) < 4):
        print("Usage:")
        print("  testClassifiers.py <inputFile> <outputFile> <classifierIdentifier>")
        print("")
        print("  <inputFile> contains a list of .inkml filenames")
        print("  <outputFile> output csv file containing UI, Top10 recognitions")
        print("  <classifierIdentifier> is one of the following:")
        print("                                                   kdtreeSymbols")
        print("                                                   kdtreeCombined")
        print("                                                   randomForestSymbols")
        print("                                                   randomForestCombined")
        print("")
        return

    # Extract the input file
    inputFile = sys.argv[1]

    # Extract the output file
    outputFile = sys.argv[2]

    # Extract the classifier identifier
    classifierIdentifier = sys.argv[3]

    print("Input:", inputFile, ", Output:", outputFile, ", classifier:", classifierIdentifier)

    # Pickled Models...
    encoderPath = "encoder_rf.pkl"
    rfModelPath = "pickle_rf.pkl"
    kdModelPath = "pickle_kd.pkl"
    encoderBothPath = "encoder_both_rf.pkl"
    rfModelBothPath = "pickle_both_rf.pkl"
    kdModelBothPath = "pickle_both_kd.pkl"

    junkPath = './trainingJunk/junk_GT_test.txt'
    # ./ trainingSymbols / iso_GT_test.txt
    # Generate the feature stack based upon the inkml files in the inputFile
    uiStack, featureStack, targetStack = getFeatures(inputFile)
    uiStackJ, featureStackJ, targetStackJ = getFeatures(junkPath)

    trainJunkSymbols = np.concatenate((featureStack, featureStackJ), axis=0)
    targetJunkSymbols = np.concatenate((targetStack, targetStackJ), axis=0)
    uicombine = np.concatenate((uiStack, uiStackJ), axis=0)


    if "randomForestSymbols" in classifierIdentifier:
        test_model(featureStack, targetStack, uiStack, encoderPath, rfModelPath, outputFile)
    if "randomForestCombined" in classifierIdentifier:
        # test_model(featureStack, targetStack, uiStack, encoderBothPath, rfModelBothPath, outputFile)
        test_model(trainJunkSymbols, targetJunkSymbols, uicombine, encoderBothPath, rfModelBothPath, outputFile)
    if "kdtreeSymbols" in classifierIdentifier:
        test_model(featureStack, targetStack, uiStack, encoderPath, kdModelPath, outputFile)
    if "kdtreeCombined" in classifierIdentifier:
        # test_model(featureStack, targetStack, uiStack, encoderBothPath, kdModelBothPath, outputFile)
        test_model(trainJunkSymbols, targetJunkSymbols, uicombine, encoderBothPath, kdModelBothPath, outputFile)

    return

main()