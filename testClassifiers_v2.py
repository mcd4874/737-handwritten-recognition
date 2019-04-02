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
        print("i=", i, ", id=", id)
    targetClasses = np.array(targetClasses, dtype=np.dtype('a16'))
    # print(targetClasses.shape)
    # print(stack.shape)
    return stack, targetClasses, uiStack

def testKDTreeClassifier(testSamplesFile, labelTestTarget, kdtree, labels, uiStack):
    predict = kdtree.predict(testSamplesFile[:, 0:])

    # only display recognition rates if labelTestTarget is provided
    if len(labelTestTarget) > 0:
        print("rf prediction: ", predict)
        print("rf actual label: ", labelTestTarget)
        print("Manual confusion matrix:")

        matrix = confusion_matrix(labelTestTarget, predict, labels=None)

        # Print Column Headers for confusion matrix
        print("", end=",")  # Initial spacer for header row
        for i in range(len(matrix[0])):
            print(labels[i], end=",")
        print("")
        for i in range(len(matrix)):
            # Prubt Row Headers for confusion matrix
            print(labels[i], end=",")
            for j in range(len(matrix[i])):
                print(matrix[i][j], end=",")
            print("")
        print(confusion_matrix(labelTestTarget, predict, labels=None))
        print(classification_report(labelTestTarget, predict, target_names=labels))
    return

def cleanString(string):
    elements = str(string).split("'")
    return elements[1];

def testRandomForestClassifier(testSamplesFile, labelTestTarget, rf, labels, uiStack):
    #The [:, 1:] is needed to skip "nan" ui first column...
    #predict = rf.predict(testSamplesFile[:, 1:])
    predict = rf.predict(testSamplesFile[:, 0:])

    combine = np.column_stack((predict, labelTestTarget))
    uniquesLabelSets = np.unique(combine,axis=0)
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
            print (labels[i], end=",")
        print("")
        for i in range(len(matrix)):
            # Prubt Row Headers for confusion matrix
            print (labels[i], end=",")
            for j in range(len(matrix[i])):
                print (matrix[i][j], end=",")
            print("")
        print(confusion_matrix(labelTestTarget, predict, labels=None))
        print(classification_report(labelTestTarget, predict, target_names=labels))
    return

def generateLabelsEncoder(targetClasses):
    enc = LabelEncoder()
    enc.fit(targetClasses)
    return enc

def transformLabels(targetClasses,encoderModel):
    return encoderModel.transform(targetClasses)

def inverseTransformLabels(targetClasses,encoderModel):
    return encoderModel.inverse_transform(targetClasses)

def get_list_indices_predict(testSamplesFile, classifier, encoderModel):
    resultList = []

    # predict = kdtree.predict(testSamplesFile)
    # No longer need to skip the first column containing nan UI
    #predict_probs = classifier.predict_proba(testSamplesFile[:, 1:])
    predict_probs = classifier.predict_proba(testSamplesFile[:, 0:])

    # print("sahpe of probs : ", predict_probs.shape)
    #sort_predict_probs_indices = predict_probs.argsort(axis = 1)[:,-10:]
    sort_predict_probs_indices = predict_probs.argsort(axis = 1)[:,-10:]

    # print(sort_predict_probs_indices)
    # print("sort probs shape:",sort_predict_probs_indices.shape)
    result = np.chararray(sort_predict_probs_indices.shape, itemsize=20)
    # predict_label = np.chararray(predict.shape, itemsize=20)
    for i in range(result.shape[0]):
        resultVector = []
        # predict_label[i] = encoderModel.classes_[predict[i]]
        for j in range(result.shape[1]):
            result[i][j] = encoderModel.classes_[sort_predict_probs_indices[i][j]]
            resultVector.append(cleanString(result[i][j]))
        result[i] = result[i][::-1]

        # Reverese the results so it is in decreasing accuracy order...
        resultVector.reverse()
        resultList.append(resultVector)
        #print("ResultVector = ", resultVector)
        #resultList.append(resultVector.reverse())
    # print(result[980:1000])
    # print(predict_label[980:1000])
    # result.append(encoderModel.classes_[sort_predict_probs_indices[i]])

    #return result
    return resultList

def mainOLD():
    testSymbols, testTargetSymbols, uiStack = stackFeatures("./symbolStack.csv", "./trainingSymbols/iso_GT_test.txt")

    encoderModel = generateLabelsEncoder(testTargetSymbols)
    # encoderModel = generateLabelsEncoder(testTargetSymbols)

    print(encoderModel.classes_)
    labelTestTarget = transformLabels(testTargetSymbols,encoderModel)
    print("test target: ",labelTestTarget)

    # Load the kdtree from our pickle
    print("Loading kdtree from pickle...")
    pkl_filename = "pickle_kdtree.pkl"
    with open(pkl_filename, 'rb') as file:
        kdtree = pickle.load(file)
    print("Finished loading.")

    #testKDTreeClassifier(testSymbols, testTargetSymbols, kdtree, encoderModel, uiStack)
    #list_test_best_label_predict = get_list_indices_predict(testSymbols, rf, encoderModel)
    #df1 = pd.DataFrame(uiStack)
    #df2 = pd.DataFrame(list_test_best_label_predict)
    #r = pd.concat([df1,df2],axis=1)
    #r.to_csv("report_table_kdtree.csv",index = False,header = False, encoding="utf-8", quoting=csv.QUOTE_NONE)

    # Load the RandomForestClassifier from our pickle
    # https://stackabuse.com/scikit-learn-save-and-restore-models/
    print("Loading random forest from pickle...")
    pkl_filename = "pickle_rf.pkl"
    with open(pkl_filename, 'rb') as file:
        rf = pickle.load(file)
    print("Finished loading.")

    testRandomForestClassifier(testSymbols, testTargetSymbols, rf, encoderModel, uiStack)
    list_test_best_label_predict = get_list_indices_predict(testSymbols, rf, encoderModel)
    df1 = pd.DataFrame(uiStack)
    df2 = pd.DataFrame(list_test_best_label_predict)
    r = pd.concat([df1,df2],axis=1)
    r.to_csv("report_table_rf.csv",index = False,header = False, encoding="utf-8", quoting=csv.QUOTE_NONE)
    return

def test_model(testSymbols, labelTestTarget, stack, encoderPath, modelPath, outputFile):
    with open(encoderPath, 'rb') as file:
        encoderModel = pickle.load(file)
    print("Finished loading encoder.")
    print(encoderModel.classes_)

    labelTestTarget = transformLabels(labelTestTarget, encoderModel)






    # Load the kdtree from our pickle
    # print("Loading kdtree from pickle...")
    # with open(modelPath, 'rb') as file:
    #     kdtree = pickle.load(file)
    # print("Finished loading.")

    # testKDTreeClassifier(testSymbols, testTargetSymbols, kdtree, encoderModel, uiStack)
    # list_test_best_label_predict = get_list_indices_predict(testSymbols,labelTestTarget, rf, encoderModel)
    # df1 = pd.DataFrame(uiStack)
    # df2 = pd.DataFrame(list_test_best_label_predict)
    # r = pd.concat([df1,df2],axis=1)
    # r.to_csv("report_table_kdtree.csv",index = False,header = False, encoding="utf-8", quoting=csv.QUOTE_NONE)

    # Load the RandomForestClassifier from our pickle
    # https://stackabuse.com/scikit-learn-save-and-restore-models/
    print("Loading model from pickle...")
    with open(modelPath, 'rb') as file:
        rf = pickle.load(file)
    print("Finished loading.")

    testRandomForestClassifier(testSymbols, labelTestTarget, rf, labelTags, stack)
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

    # Generate the feature stack based upon the inkml files in the inputFile
    uiStack, featureStack, targetStack = getFeatures(inputFile)

    # Predict using randomforest
    if "randomForestSymbols" in classifierIdentifier:
        test_model(featureStack, targetStack, uiStack, encoderPath, rfModelPath, outputFile)
    if "randomForestCombined" in classifierIdentifier:
        test_model(featureStack, targetStack, uiStack, encoderBothPath, rfModelBothPath, outputFile)
    if "kdtreeSymbols" in classifierIdentifier:
        test_model(featureStack, targetStack, uiStack, encoderPath, kdModelPath, outputFile)
    if "kdtreeCombined" in classifierIdentifier:
        test_model(featureStack, targetStack, uiStack, encoderBothPath, kdModelBothPath, outputFile)

    return

main()