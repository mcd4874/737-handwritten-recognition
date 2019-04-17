#
# CSCI-737: Project 1
# Authors: Eric Hartman and William Duong
#
from sklearn.preprocessing import MinMaxScaler
import numpy as np
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

# Loads the sample data from the given filename into the provided dictionary
def loadSampleIntoDictionary(idToUILookup, dictionary, filename):
    # print("filename: ", filename)
    tree = ET.parse(filename)
    root = tree.getroot()
    id = None
    strokes = []

    # extract annotation type
    for child in root.iter():
        # print("child.tag: ", child.tag, ",  child.attrib: ", child.attrib);
        if child.tag == '{http://www.w3.org/2003/InkML}annotation' and child.attrib['type'] == 'UI':
            # Retrieve the unique identifier for this symbol
            UI = child.text

            # Take the last _'th split as the unique identifier for this sample
            elements = UI.split("_")
            id = elements[len(elements) - 1]
            print("filename = ", filename, ", value = ", child.text, ", type = ", child.attrib['type'])

            # Add the id to UI to our lookup dictionary
            idToUILookup[id] = UI

        if child.tag == '{http://www.w3.org/2003/InkML}trace':
            # Retrieve the stroke(s) for this symbol
            # print("value = ", child.text)
            strokes.append(child.text)

    # Convert strokes into 3D numpy array: [stroke][x][y]
    strokes3d = parseStrokesInto3DArray(strokes)

    # Add symbol to our dictionary
    dictionary[UI] = strokes3d
    return

# Generates a cache in "uiToFilename" dictionary based on UI -> Filename mapping for
# all inkml files in the given directoryPath
def cacheUItoFilename(uiToFilename, directoryPath):

    # Iterate over all of the files
    fileList = os.listdir(directoryPath)
    for filename in fileList:
        if "inkml" in filename:
            tree = ET.parse(directoryPath + "/" + filename)
            root = tree.getroot()

            # extract annotation type
            for child in root.iter():
                #print("child.tag: ", child.tag, ",  child.attrib: ", child.attrib);
                if child.tag == '{http://www.w3.org/2003/InkML}annotation' and child.attrib['type'] == 'UI':
                    # Retrieve the unique identifier for this symbol
                    UI=child.text
                    uiToFilename[UI] = directoryPath + filename
                    print("caching filename: ", directoryPath + filename, " and UI: ", UI)

    return

# Generates sample images with dots for the stroke coordinates
def generateNoConnectImages(dictionary, imageDictionary, w, h):
    size = (h+1, w+1, 1)
    keys = list(dictionary)
    for i in range(0,len(dictionary)):
        id = keys[i]
        print("NoConnect: Processing image for ", id)
        img = np.full(size, 255, np.uint8)

        # Process each point
        for strokeCoordinate in dictionary[id]:
            # Stroke, X, Y into img[Y][X]
            #print("strokeCoordinate=", strokeCoordinate)
            img[math.floor(strokeCoordinate[2])][math.floor(strokeCoordinate[1])] = 0

        # Add the unraveled image to the image dictionary
        imageDictionary[id] = img.ravel()
    return

# Generates sample images with lines between consecutive stroke coordinates
def generateConnectedImages(dictionary, imageDictionary, w, h, thickness):
    size = (h+1, w+1, 1)

    keys = list(dictionary)
    for i in range(0,len(dictionary)):
        id = keys[i]

        #print("Connected: Processing image for ", id)
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

        # Add the unraveled image to the image dictionary
        imageDictionary[id] = img.ravel()

        #filename = "" + id + ".png"
        #cv2.imwrite(path + "/" + filename, img)

    return

# Generates a Features stack ordered by id=0, 1, 2, 3, ...
def generateFeatureStack(idToUILookup, dictionary):
    stack = []
    uiStack = []

    keys = list(dictionary)
    for i in range(0,len(dictionary)):
        id = keys[i]
        #print("Processing id=", id)
        features = []

        # Add the UI to the uiStack
        #uiStack.append(idToUILookup[id])
        uiStack.append(id)

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

            # Convert negative to positive angles
            if angle < 0.0:
                angle = angle + 360
            features.append(angle)
            #print("angle=", angle)

            # Remember this as last coordinate
            lastCoordinate = strokeCoordinate

        #print("features=", features)
        stack.append(features)

    return uiStack, stack

# Normalizes the angles to the nearest sector
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
        #print("orientation normalized feature=",newFeature)

    return normalizedStack

# Reduces sequences of duplicate angles to just one instance of the angle
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
        #print("deduplicate feature=",newFeature)

    return normalizedStack

# Saves the output in CSV form for debugging purposes
def saveCSV(uiStack, stack, filename):
    file = open(filename, "w+")

    for f in range(len(stack)):
        features = stack[f]

        # First, write out the UI as the first column
        file.write('' + str(uiStack[f]))
        file.write(",")

        # Then, write out all of the features in the stack
        for i in range(len(features)):
            file.write('' + str(features[i]))
            if (i+1) < (len(features)):
                file.write(",")
        file.write("\n")

    file.close()

# Normalizes the number of angles to the given featureCount
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

# Appends the orientation angle histogram bins to the feature vectors
def appendBinFeatures(stack, binCount):
    newStack = []

    for features in stack:
        newFeatures = features.copy()

        # Initialize bin features
        binFeatures = []
        for i in range(binCount):
            binFeatures.append(0.0)

        # Walk through each actual feature to populate the bins
        for i in range(len(newFeatures)):
            angle = newFeatures[i]
            if angle == 0:
                # Skip 0 angles, as these are most likely "filler" for sequences that arent long enough for featuresCount
                # and therefore do not represent the underlying shape of the symbol.
                continue
            #if angle < 0:
            #    angle = angle + 360.0
            # Add small epsilon to force angle=360 into last bin instead of out-of-range
            sectorSize = 360.0 / binCount  + 0.01
            chosenBin = int(math.floor(angle / sectorSize))
            #print("chosenBin = ", chosenBin, ", i=", i, ", newFeatures=", newFeatures)
            binFeatures[chosenBin] += 1.0

        newFeatures.extend(binFeatures)
        #print("bins: ", binFeatures)
        #print("New features with bins: ", newFeatures)
        newStack.append(newFeatures)

    return newStack

# Appends the unraveled image pixel values to the feature vectors
def appendImageFeatures(stack, imageDictionary):
    newStack = []

    keys = list(imageDictionary)
    for i in range(len(stack)):
        id = keys[i]
        #print("Appending image features for id=", id)
        features = stack[i]
        newFeatures = features.copy()

        # Obtain the raveled image
        imageFeatures = imageDictionary[id]

        newFeatures.extend(imageFeatures)
        #print("imageFeatures: ", imageFeatures)
        #print("New features with imageFeatures: ", newFeatures)
        newStack.append(newFeatures)

    return newStack

# Get the aspect ratio as height/width as a float
def getAspectRatio(strokes):
    aspectRatio = 1.0

    # Extremities
    upperLeftX = 99999.0
    upperLeftY = 99999.0
    lowerRightX = -99999.0
    lowerRightY = -99999.0

    # Iterate over stroke coordinate and record the upper left and lower right extrema points
    for strokeCoordinate in strokes:
        if strokeCoordinate[1] < upperLeftX:
            upperLeftX = strokeCoordinate[1]
        if strokeCoordinate[1] > lowerRightX:
            lowerRightX = strokeCoordinate[1]
        if strokeCoordinate[2] < upperLeftY:
            upperLeftY = strokeCoordinate[2]
        if strokeCoordinate[2] > lowerRightY:
            lowerRightY = strokeCoordinate[2]

    # Calculate aspect ratio
    width = abs(lowerRightX - upperLeftX)
    height = abs(lowerRightY - upperLeftY)

    # Protection against divide-by-zero errors
    if width < 0.01:
        aspectRatio = 1.0
    else:
        aspectRatio = float(height)/float(width)

    return aspectRatio

# Appends the Project 2 enhanced features to the feature vectors
def appendProject2Features(stack, symbolsDictionary):
    newStack = []

    keys = list(symbolsDictionary)
    for i in range(len(stack)):
        id = keys[i]
        #print("Appending Project 2 features for id=", id)
        features = stack[i]
        newFeatures = features.copy()

        # Initialize project2 features
        project2Features = []

        # Add number of strokes as a feature
        strokesData = symbolsDictionary[id]
        i, j = strokesData.shape
        strokeCount = strokesData[i-1][0] + 1
        project2Features.append(strokeCount)

        # Add aspect ratio as a feature
        aspectRatio = getAspectRatio(strokesData)
        project2Features.append(aspectRatio)

        newFeatures.extend(project2Features)
        #print("project2features: ", project2Features)
        #print("New features with project2features: ", newFeatures)
        newStack.append(newFeatures)

    return newStack

# Utility method called by testClassifiers_v2 to retrieve all of the features for the
# samples denoted in the given input file
def getFeatures(inputFile):
    w = 20
    h = 20
    thickness = 4
    sector = 45
    binCount = 8
    featureCount = 30

    # Initialize the lists
    symbolsDictionary = dict()
    symbolsImageDictionary = dict()
    symbolIdToUILookup = dict()

    # Project 2 extension...
    symbolsDictionaryNotNormalized = dict()

    # Load the input file
    data = np.genfromtxt(inputFile, delimiter=',', dtype=str)

    # Construct the target stack (if its available)
    targetStack = []

    for i in range(len(data)):
        #print("Dimensions = ", len(data.shape))
        if len(data.shape) > 1:
            print("input: ", data[i][0])
            loadSampleIntoDictionary(symbolIdToUILookup, symbolsDictionary, data[i][0])
            loadSampleIntoDictionary(symbolIdToUILookup, symbolsDictionaryNotNormalized, data[i][0])
        else:
            print("input: ", data[i])
            loadSampleIntoDictionary(symbolIdToUILookup, symbolsDictionary, data[i])
            loadSampleIntoDictionary(symbolIdToUILookup, symbolsDictionaryNotNormalized, data[i])

        if len(data.shape) > 1:
            # Binarize the target stack
            print("target: ", data[i][1])
            targetStack.append(bytes(data[i][1], encoding='utf-8'))

    # Perform normalization to all samples in our dictionary
    normalizeSamples(symbolsDictionary, w, h)

    # Create "no-connect" images
    #generateNoConnectImages(symbolsDictionary, symbolsImageDictionary, w, h);

    # Create "connected" images
    generateConnectedImages(symbolsDictionary, symbolsImageDictionary, w, h, thickness)

    symbolUIStack, symbolStack = generateFeatureStack(symbolIdToUILookup, symbolsDictionary)
    symbolStack = orientationNormalization(symbolStack, sector)
    symbolStack = deduplicationNormalization(symbolStack)
    symbolStack = featureCountNormalization(symbolStack, featureCount)
    symbolStack = appendBinFeatures(symbolStack, binCount)
    #print("stack shape after append bin features=", len(symbolStack[0]))
    symbolStack = appendImageFeatures(symbolStack, symbolsImageDictionary)
    #print("stack shape after append image features=", len(symbolStack[0]))

    # Project 2 extension...
    symbolStack = appendProject2Features(symbolStack, symbolsDictionaryNotNormalized)

    #saveCSV(symbolUIStack, symbolStack, "./symbolStack.csv")

    return symbolUIStack, np.array(symbolStack), targetStack;

# Utility method called by segmenter to retrieve all of the features for the
# given subset of strokes provided via subsetIdentifier and strokes
def getFeaturesForStrokes(subsetIdentifier, strokes):
    w = 20
    h = 20
    thickness = 4
    sector = 45
    binCount = 8
    featureCount = 30

    # Initialize the lists
    symbolsDictionary = dict()
    symbolsImageDictionary = dict()
    symbolIdToUILookup = dict()

    # Project 2 extension...
    symbolsDictionaryNotNormalized = dict()

    # Construct the target stack (if its available)
    targetStack = []

    # Load sample into dictionary:
    #  Instead of parsing the XML file, do the basic
    #  activities that were done in loadSampleIntoDictionary()

    # Trivial use of symbolsIdToUILookup
    symbolIdToUILookup[0] = subsetIdentifier

    # Convert strokes into 3D numpy array: [stroke][x][y]
    strokes3d = parseStrokesInto3DArray(strokes)

    # Add symbol to our dictionary
    symbolsDictionary[subsetIdentifier] = strokes3d

    # Project 2 extension...
    symbolsDictionaryNotNormalized[subsetIdentifier] = strokes3d

    # Perform normalization to all samples in our dictionary
    normalizeSamples(symbolsDictionary, w, h)

    # Create "no-connect" images
    #generateNoConnectImages(symbolsDictionary, symbolsImageDictionary, w, h);

    # Create "connected" images
    generateConnectedImages(symbolsDictionary, symbolsImageDictionary, w, h, thickness)

    symbolUIStack, symbolStack = generateFeatureStack(symbolIdToUILookup, symbolsDictionary)
    symbolStack = orientationNormalization(symbolStack, sector)
    symbolStack = deduplicationNormalization(symbolStack)
    symbolStack = featureCountNormalization(symbolStack, featureCount)
    symbolStack = appendBinFeatures(symbolStack, binCount)
    #print("stack shape after append bin features=", len(symbolStack[0]))
    symbolStack = appendImageFeatures(symbolStack, symbolsImageDictionary)
    #print("stack shape after append image features=", len(symbolStack[0]))

    # Project 2 extension...
    symbolStack = appendProject2Features(symbolStack, symbolsDictionaryNotNormalized)

    #saveCSV(symbolUIStack, symbolStack, "./symbolStack.csv")

    return symbolUIStack, np.array(symbolStack), targetStack