from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import unique_labels
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import xml.etree.ElementTree as ET
import os
import cv2
from yellowbrick.classifier import ConfusionMatrix
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
    identifier = list()
    for i in range(sampleSize):
        # Extract the target class
        target = data[i][1]
        targetClasses.append(target)
        # Extract the unique identifier for the symbol
        elements = str(data[i][0]).split("_")
        identifier.append(str(data[i][0]))
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
    identifier = np.array(identifier, dtype=np.dtype('a16'))
    targetClasses = np.array(targetClasses, dtype=np.dtype('a16'))
    # print(targetClasses.shape)
    # print(stack.shape)
    return stack,targetClasses,identifier

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

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

import pandas as pd
def main():
    trainSymbols,targetSymbols,trainIdentifiers = stackImages("./images/connect/symbols/", "./trainingSymbols/iso_GT_train.txt")
    testSymbols,testTargetSymbols,testIdentifiers = stackImages("./images/connect/symbols/", "./trainingSymbols/iso_GT_test.txt")

    print(testIdentifiers.shape)
    print(testIdentifiers[0])
    print(np.isnan(testSymbols).any())
    encoderModel = generateLabelsEncoder(targetSymbols)
    saveModel("encoder.pkl",encoderModel)

    # encoderModel = loadModel("encoder.pkl")

    # trainSymbols = trainSymbols[:20000]
    # targetSymbols = targetSymbols[:20000]
    # testSymbols = testSymbols[:1000]
    # testTargetSymbols = testTargetSymbols[:1000]
    labelTrainTarget = transformLabels(targetSymbols,encoderModel)
    labelTestTarget = transformLabels(testTargetSymbols,encoderModel)

    # kdtree = trainKDTreeClassifier(trainSymbols,labelTrainTarget)
    # saveModel("result.pkl",kdtree)
    kdtree = loadModel("result.pkl")
    list_test_best_label_predict = get_list_indices_predict(testSymbols,labelTestTarget, kdtree,encoderModel)
    # print(list_test_best_label_predict.shape)
    #
    df1 = pd.DataFrame(testIdentifiers)
    df2 = pd.DataFrame(list_test_best_label_predict)
    r = pd.concat([df1,df2],axis=1)
    r.to_csv("report_table.csv",index = False,header = False)

    # Plot non-normalized confusion matrix
    # plot_confusion_matrix(kdtree.predict(testSymbols), labelTestTarget, classes=encoderModel.classes_,
    #                       title='Confusion matrix, without normalization')
    # plt.show()
    # iris_cm = ConfusionMatrix(
    #     kdtree, classes=encoderModel.classes_
    #
    # )
    #
    # iris_cm.fit(trainSymbols, labelTrainTarget)
    # iris_cm.score(testSymbols, labelTestTarget)
    # #
    # iris_cm.poof()
    # plt.show()

    # testKDTreeClassifier(testSymbols,labelTestTarget, kdtree,encoderModel)
    # maxTrees = 100
    # maxDepth = 20
    # print("random forest ::")
    # rf = trainRandomForestClassifier(trainSymbols, targetSymbols, maxTrees, maxDepth)
    # testRandomForestClassifier(testSymbols, testTargetSymbols, rf, encoderModel)

    return

main()