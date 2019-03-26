from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import math
import xml.etree.ElementTree as ET
import os
import cv2


# Function to normalize the (x,y) coordinates contained within the given dictionary according to
#  width and heights provided.  This normalization is performed on a per-sample basis.
#  (x,y) is normalized to (0->w, 0->h)
def normalizeSamples(dictionary, w, h):
    # Setup our min-max scalers independently for X and Y coordinates
    scalerX = MinMaxScaler(feature_range=(0, w), copy=False)
    scalerY = MinMaxScaler(feature_range=(0, h), copy=False)

    # Iterate over each key in the dictionary
    for id in dictionary:
        #print("id = ", id)
        strokes = dictionary[id]
        # strokes is a 3-d array such that: strokeNumber, X, Y

        # Apply the MinMax scaler to our X and Y features
        scalerX.fit(strokes[:,1:2])
        scalerX.transform(strokes[:,1:2])
        scalerY.fit(strokes[:,2:3])
        scalerY.transform(strokes[:,2:3])
        #print("Transformed strokes=", strokes)

    return

# CSV like
# Stroke#, X, Y
# Stroke#, X, Y
# ...
def parseStrokesInto3DArray(strokes):
    # Initialize strokes array...
    strokesArray = np.empty((0,3))
    strokesCounter = 0
    for stroke in strokes:
        # Delimit with comma
        coordinateStrings = stroke.strip().split(',')
        #print("Coordinates in stroke = ", len(coordinateStrings))
        strokeArray = np.full((len(coordinateStrings), 3), 0.0, dtype=float)

        # Convert coordinateStrings
        coordinateCounter = 0
        for coordinateString in coordinateStrings:
            #print("coordinateString=[", coordinateString, "]")
            # Delimit with space
            coordinates = coordinateString.strip().split(' ')
            #print("(x,y) = (" + coordinates[0], ",", coordinates[1], ")")
            strokeArray[coordinateCounter][0] = strokesCounter
            strokeArray[coordinateCounter][1] = coordinates[0]
            strokeArray[coordinateCounter][2] = coordinates[1]
            coordinateCounter += 1

        #print("Stroke np array = ", strokeArray)
        strokesArray = np.append(strokesArray, strokeArray, axis=0)

        #print("Strokes np array = ", strokesArray)
        strokesCounter += 1

    #print("StrokesArray = ", strokesArray)

    return strokesArray

def loadSamplesIntoDictionary(dictionary, directoryPath, limit):
    counter = 0

    # Iterate over all of the files
    fileList = os.listdir(directoryPath)
    for filename in fileList:
        if "inkml" in filename:
            #print("filename: ", filename)
            tree = ET.parse(directoryPath + "/" + filename)
            root = tree.getroot()
            id=None
            strokes=[]

            # extract annotation type
            for child in root.iter():
                #print("child.tag: ", child.tag, ",  child.attrib: ", child.attrib);
                if child.tag == '{http://www.w3.org/2003/InkML}annotation' and child.attrib['type'] == 'UI':
                    # Retrieve the unique identifier for this symbol
                    id=child.text

                    # Take the last _'th split as the unique identifier for this sample
                    elements = id.split("_")
                    id = elements[len(elements)-1]
                    print("filename = ", filename, ", value = ", child.text, ", type = ", child.attrib['type'])

                if child.tag == '{http://www.w3.org/2003/InkML}trace':
                    # Retrieve the stroke(s) for this symbol
                    #print("value = ", child.text)
                    strokes.append(child.text)

            # Convert strokes into 3D numpy array: [stroke][x][y]
            strokes3d = parseStrokesInto3DArray(strokes)

            # Add symbol to our dictionary
            dictionary[id] = strokes3d
            counter += 1

            # Premature break for testing purposes
            if limit > 0 and counter >= limit:
                break
            # break

    return

def generateNoConnectImages(dictionary, path, w, h):
    size = (h+1, w+1, 1)
    for id in dictionary:
        print("NoConnect: Processing image for ", id)
        img = np.full(size, 255, np.uint8)

        # Process each point
        for strokeCoordinate in dictionary[id]:
            # Stroke, X, Y into img[Y][X]
            #print("strokeCoordinate=", strokeCoordinate)
            img[math.floor(strokeCoordinate[2])][math.floor(strokeCoordinate[1])] = 0
        filename = "" + id + ".png"
        cv2.imwrite(path + "/" + filename, img)

    return


def generateConnectedImages(dictionary, path, w, h, thickness):
    size = (h+1, w+1, 1)
    for id in dictionary:
        print("Connected: Processing image for ", id)
        img = np.full(size, 255, np.uint8)

        # Process each point
        newStroke = True
        strokeId = 0
        lastCoordinate = None
        for strokeCoordinate in dictionary[id]:
            if newStroke:
                newStroke = False
                lastCoordinate = strokeCoordinate
                continue

            if strokeId != strokeCoordinate[0]:
                strokeId = strokeCoordinate[0]
                lastCoordinate = strokeCoordinate
                continue

            # Connect from prior coordinate to current coordinate
            cv2.line(img, (math.floor(lastCoordinate[1]), math.floor(lastCoordinate[2])),
                     (math.floor(strokeCoordinate[1]), math.floor(strokeCoordinate[2])),
                     (0,0,0), thickness)
            #print("Drawing line from (", math.floor(lastCoordinate[1]), ",", math.floor(lastCoordinate[2]), ") to (", math.floor(strokeCoordinate[1]), ", ",
            #      math.floor(strokeCoordinate[2]), ")")

            # Remember this as last coordinate
            lastCoordinate = strokeCoordinate

        filename = "" + id + ".png"
        cv2.imwrite(path + "/" + filename, img)

    return

def generateFeatureStack(dictionary):
    stack = []
    for id in dictionary:
        print("Processing id=", id)
        features = []

        # Process each point
        newStroke = True
        strokeId = 0
        lastCoordinate = None
        for strokeCoordinate in dictionary[id]:
            if newStroke:
                features.append(-360.0)
                newStroke = False
                lastCoordinate = strokeCoordinate
                continue

            if strokeId != strokeCoordinate[0]:
                strokeId = strokeCoordinate[0]
                lastCoordinate = strokeCoordinate
                features.append(-360.0)
                continue

            # Calculate the angle of the line connecting lastCoordinate[1] to lastCoordinate[2]
            angle = math.degrees(math.atan2(strokeCoordinate[2] - lastCoordinate[2], strokeCoordinate[1] - lastCoordinate[1]))
            features.append(angle)
            #print("angle=", angle)

            # Remember this as last coordinate
            lastCoordinate = strokeCoordinate

        print("features=", features)
        stack.append(features)

    return stack

def orientationNormalization(stack, sector):
    normalizedStack = []
    for features in stack:
        newFeature = []
        for i in range(0,len(features)):
            feature = features[i]
            # Round the feature to the nearest "15"
            approximateAngle = float(float(feature) / float(sector))
            approximateAngle = round(approximateAngle) * sector
            #print("feature=", feature, ", approximateAngle=", approximateAngle)
            newFeature.append(approximateAngle)
        normalizedStack.append(newFeature)
        print("orientation normalized feature=",newFeature)

    return normalizedStack

def deduplicationNormalization(stack):
    normalizedStack = []
    for features in stack:
        newFeature = []
        lastFeature = features[0]

        # Always retain the first feature
        newFeature.append(lastFeature)

        # Only append differences
        for i in range(1, len(features)):
            feature = features[i]
            if feature == lastFeature:
                continue

            # Found a difference, save it!
            newFeature.append(feature)

            # Update lastFeature to current feature
            lastFeature = feature

        normalizedStack.append(newFeature)
        print("deduplicate feature=",newFeature)

    return normalizedStack

def saveCSV(stack, filename):
    file = open(filename, "w+")

    for features in stack:
        for i in range(len(features)):
            file.write('' + str(features[i]))
            if (i+1) < (len(features)):
                file.write(",")
        file.write("\n")

    file.close()


def featureCountNormalization(stack, featureCount):
    newStack = []

    for features in stack:
        newFeatures = features[:featureCount]
        missingFeatures = featureCount - len(newFeatures)
        for i in range(missingFeatures):
            newFeatures.append(0.0)
        newStack.append(newFeatures)
        #print("missing=", missingFeatures, ", featureCountNormalization=", newFeatures)

    return newStack

def main():
    limit = 0
    w = 15
    h = 15
    thickness = 2
    sector = 15.0
    featureCount = 25

    # Initialize the dictionary
    symbolsDictionary = dict()
    junkDictionary = dict()

    # Load Symbols and Junk samples into the common dictionary, indexed by UI "unique identifier?"
    loadSamplesIntoDictionary(symbolsDictionary, "./trainingSymbols/", limit)
    loadSamplesIntoDictionary(junkDictionary, "./trainingJunk/", limit)

    # Perform normalization to all samples in our dictionary
    normalizeSamples(symbolsDictionary, w, h)
    normalizeSamples(junkDictionary, w, h)

    # Create "no-connect" images
    #generateNoConnectImages(symbolsDictionary, "./images/no_connect/symbols/", w, h)
    #generateNoConnectImages(junkDictionary, "./images/no_connect/junk/", w, h)

    # Create "connected" images
    #generateConnectedImages(symbolsDictionary, "./images/connect/symbols/", w, h, thickness)
    #generateConnectedImages(junkDictionary, "./images/connect/junk/

    symbolStack = generateFeatureStack(symbolsDictionary)
    symbolStack = orientationNormalization(symbolStack, sector)
    symbolStack = deduplicationNormalization(symbolStack)
    symbolStack = featureCountNormalization(symbolStack, featureCount)

    junkStack = generateFeatureStack(junkDictionary)
    junkStack = orientationNormalization(junkStack, sector)
    junkStack = deduplicationNormalization(junkStack)
    junkStack = featureCountNormalization(junkStack, featureCount)

    #print("stack=", stack)

    saveCSV(symbolStack, "./symbolStack.csv")
    saveCSV(junkStack, "./junkStack.csv")
    return

main()