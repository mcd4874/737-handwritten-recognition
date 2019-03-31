from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
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
    sampleSize = len(data)
    print(sampleSize)
    # sampleSize = 100
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
        # print("filename=",  filename)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        #print("img =", img)
        #print("img type =", type(img))

        imgRavel = img.ravel()
        # print(imgRavel.shape)
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
    # print("About to run KDTree")
    # kdtree = KDTree(stack)
    # print("About to query KDTree")
    # dist, ind = kdtree.query(stack[1], k=3)
    # print("indices of 3 closest = ", ind)
    # print("distances of 3 closest = ", dist)

    kdTree = KNeighborsClassifier(n_neighbors=1, algorithm='kd_tree')
    kdTree.fit(stack,targetClasses)

    return kdTree

def trainRandomForestClassifier(stack, targetClasses, maxTrees, maxDepth):
    rf = RandomForestClassifier(n_estimators=maxTrees, max_depth=maxDepth)
    rf.fit(stack,targetClasses)
    return rf

def get_list_indices_predict(testSamplesFile, labelTestTarget, kdtree, encoderModel):
    # predict = kdtree.predict(testSamplesFile)
    predict_probs = kdtree.predict_proba(testSamplesFile)
    # print("sahpe of probs : ", predict_probs.shape)
    sort_predict_probs_indices = predict_probs.argsort(axis = 1)[:,-10:]
    # print(sort_predict_probs_indices)
    # print("sort probs shape:",sort_predict_probs_indices.shape)
    result = np.chararray(sort_predict_probs_indices.shape, itemsize=20)
    # predict_label = np.chararray(predict.shape, itemsize=20)
    for i in range(result.shape[0]):
        # predict_label[i] = encoderModel.classes_[predict[i]]
        for j in range(result.shape[1]):
            result[i][j] = encoderModel.classes_[sort_predict_probs_indices[i][j]]
        result[i] = result[i][::-1]
    # print(result[980:1000])
    # print(predict_label[980:1000])
    # result.append(encoderModel.classes_[sort_predict_probs_indices[i]])
    return result


def testKDTreeClassifier(testSamplesFile, labelTestTarget, kdtree, encoderModel):
    predict = kdtree.predict(testSamplesFile)



    print("prediction: ",predict)
    print("actual label : ",labelTestTarget)
    # print('confusion matrix',confusion_matrix(labelTestTarget, predict, classes=encoderModel.classes_, label_encoder=encoder))
    print(confusion_matrix(labelTestTarget, predict))
    #print(classification_report(labelTestTarget, predict, target_names=encoderModel.classes_))
    print(classification_report(labelTestTarget, predict, target_names=encoderModel.classes_))
    return
def testRandomForestClassifier(testSamplesFile, labelTestTarget, rf, encoderModel):
    predict = rf.predict(testSamplesFile)
    print("rf prediction: ", predict)
    print("rf actual label: ", labelTestTarget)
    print(confusion_matrix(labelTestTarget, predict, labels=None))
    print(classification_report(labelTestTarget, predict, target_names=encoderModel.classes_))
    return

def generateLabelsEncoder(targetClasses):
    enc = LabelEncoder()
    enc.fit(targetClasses)
    return enc

def transformLabels(targetClasses,encoderModel):
    return encoderModel.transform(targetClasses)

def inverseTransformLabels(targetClasses,encoderModel):
    return encoderModel.inverse_transform(targetClasses)

def saveModel(pkl_filename,model):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
def loadModel(pkl_filename):
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
        print("finish load model from ",pkl_filename)
        return pickle_model



def main():
    # trainSymbols,targetSymbols = stackImages("./images/connect/symbols/", "./trainingSymbols/iso_GT_train.txt")
    testSymbols,testTargetSymbols = stackImages("./images/connect/symbols/", "./trainingSymbols/iso_GT_test.txt")

    # encoderModel = generateLabelsEncoder(targetSymbols)
    # saveModel("encoder.pkl",encoderModel)

    encoderModel = loadModel("encoder.pkl")

    # trainSymbols = trainSymbols[:20000]
    # targetSymbols = targetSymbols[:20000]
    testSymbols = testSymbols[:1000]
    testTargetSymbols = testTargetSymbols[:1000]
    # labelTrainTarget = transformLabels(targetSymbols,encoderModel)
    labelTestTarget = transformLabels(testTargetSymbols,encoderModel)

    # kdtree = trainKDTreeClassifier(trainSymbols,labelTrainTarget)
    # saveModel("result.pkl",kdtree)
    kdtree = loadModel("result.pkl")
    list_test_best_label_predict = get_list_indices_predict(testSymbols,labelTestTarget, kdtree,encoderModel)





    # testKDTreeClassifier(testSymbols,labelTestTarget, kdtree,encoderModel)
    # maxTrees = 100
    # maxDepth = 20
    # print("random forest ::")
    # rf = trainRandomForestClassifier(trainSymbols, targetSymbols, maxTrees, maxDepth)
    # testRandomForestClassifier(testSymbols, testTargetSymbols, rf, encoderModel)

    return

main()