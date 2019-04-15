#
# segmenter.py
#
# Authors: Eric Hartman and William Duong
#
# Segments a given .inkml file into an .lg file
#

import sys
import numpy as np
import xml.etree.ElementTree as ET
import itertools
import math
from sklearn.preprocessing import MinMaxScaler
from generateFeatureStack import getFeaturesForStrokes
from testClassifiers_v2 import loadModels
from testClassifiers_v2 import classifyStrokeSubset


# Extract the list of symbols from the given inkml file
def extractSymbolsFromFile(filename):
    # Initialize our list of symbols
    symbols = []

    # Define our ParseError
    ParseError = ET.ParseError

    try:
        # Parse inkml and get root
        tree = ET.parse(filename)

        # extract annotation type
        for node in tree.findall('.//{http://www.w3.org/2003/InkML}traceGroup/{http://www.w3.org/2003/InkML}traceGroup/{http://www.w3.org/2003/InkML}annotation'):
            symbols.append(node.text.strip())

        print("File: ", filename, ", symbols: ", symbols)

    except ParseError:
        print("ParseError on ", filename)

    return symbols

# Extract the mapping from strokeId to stroke from the given file
#  if mode is "train", extracts the ground truth mapping from stroke_ids to symbol_id
#                      extracts the ground truth mapping from symbol_id to symbol
def extractStrokes(filename, mode):
    # Initialize our results
    strokeIdToStroke = dict()
    gt_strokeIdToSymbolId = dict()
    gt_symbolIdToSymbol = dict()

    # Define our ParseError
    ParseError = ET.ParseError

    try:
        # Parse inkml and get root
        tree = ET.parse(filename)

        # Get all of the <trace> elements, these are the strokes
        for node in tree.findall('.//{http://www.w3.org/2003/InkML}trace'):
            strokeId = node.attrib['id']
            strokeIdToStroke[strokeId] = node.text.strip()
            #print("File: ", filename, ", strokeId=", strokeId, ", strokeData=", strokeIdToStroke[strokeId])

        # Training mode only
        if mode == 'train':

            # If mode is train, let's get the ground truth segmentation of strokes to symbols
            for node in tree.findall('.//{http://www.w3.org/2003/InkML}traceGroup/{http://www.w3.org/2003/InkML}traceGroup'):
                #print("valid attributes for this node: ", node.attrib.keys())
                symbolId = node.attrib['{http://www.w3.org/XML/1998/namespace}id']

                # Find the truth for this symbol id
                truth = tree.find(".//{http://www.w3.org/2003/InkML}traceGroup[{http://www.w3.org/2003/InkML}annotation][@{http://www.w3.org/XML/1998/namespace}id='" + symbolId + "']/")
                #print("File: ", filename, ", symbolId=", symbolId, ", truth=", truth.text.strip())
                symbol = truth.text.strip()

                # Populate our ground truth mapping
                gt_symbolIdToSymbol[symbolId] = symbol

                # Find the strokes that compose this symbol
                for traceViewNode in tree.findall(".//{http://www.w3.org/2003/InkML}traceGroup[@{http://www.w3.org/XML/1998/namespace}id='" + symbolId + "']/{http://www.w3.org/2003/InkML}traceView"):
                    #print("valid attributes for this node: ", traceViewNode.attrib.keys())
                    strokeId = traceViewNode.attrib['traceDataRef']
                    #print("Found member strokeId=", strokeId)

                    # Populate our ground truth mapping
                    gt_strokeIdToSymbolId[strokeId] = symbolId

    except ParseError:
        print("ParseError on ", filename)

    return strokeIdToStroke, gt_strokeIdToSymbolId, gt_symbolIdToSymbol

# Returns the centroid (or mean (x,y) position) for each stroke
# The strokes are Min-Max scaled to a range of [0,maxWidth] and [0,maxHeight] for x and y respectively.
def calculateCentroidsByStrokeId(strokeIdToStroke, maxWidth, maxHeight):
    strokeIdToCentroid = dict()

    # Determine the global peaks/troughs for x,y values across all strokes
    peakX = -99999.0
    peakY = -99999.0
    troughX = 99999.0
    troughY = 99999.0

    # Initialize our scaling target
    centroids = np.full((len(strokeIdToStroke.keys()), 2), 0.0, dtype=float)

    # Iterate over all of the strokes..
    index = 0
    for strokeId in strokeIdToStroke.keys():
        # Coordinates
        x = 0.0
        y = 0.0

        stroke = strokeIdToStroke[strokeId]
        coordinateStrings = str(stroke).split(',')
        for coordinateString in coordinateStrings:
            clean = coordinateString.strip()
            elements = clean.split(' ')
            xValue = float(elements[0])
            yValue = float(elements[1])
            x += xValue
            y += yValue

            # Update peaks and troughs
            if xValue > peakX:
                peakX = xValue
            if xValue < troughX:
                troughX = xValue
            if yValue > peakY:
                peakY = yValue
            if yValue < troughY:
                troughY = yValue

        # Save the centroid in the map
        centroid = [float(x)/float(len(stroke)), float(y)/float(len(stroke))]
        #print("Centroid = ", centroid)
        strokeIdToCentroid[strokeId] = centroid

        # Save the centroid in the np.array
        centroids[index][0] = centroid[0]
        centroids[index][1] = centroid[1]

        # Increment index
        index += 1

    # Min-Max scale the centroid positions according to the global peaks and troughs
    scalerX = MinMaxScaler(feature_range=(0, maxWidth), copy=False)
    scalerY = MinMaxScaler(feature_range=(0, maxHeight), copy=False)

    scalerX.fit(centroids[:, 0:1])
    scalerX.transform(centroids[:, 0:1])
    scalerY.fit(centroids[:, 1:2])
    scalerY.transform(centroids[:, 1:2])

    # Update the strokeIdToCentroid map with our scaled centroids
    index = 0
    for strokeId in strokeIdToCentroid.keys():
        strokeIdToCentroid[strokeId] = centroids[index]
        index += 1

    return strokeIdToCentroid

# Returns the total distance from the centroid of the subset to every member of the subset
# Something like a "star distance"
def calculateSumOfDistancesToCentroid(subset, strokeIdToCentroid):
    distance = 0.0

    # Calculate the centroid of the subset
    x = 0.0
    y = 0.0
    for strokeId in subset:
        strokeCentroid = strokeIdToCentroid[strokeId]
        x += strokeCentroid[0]
        y += strokeCentroid[1]

    centroidOfSubset = [float(x)/float(len(subset)), float(y)/float(len(subset))]

    # Calculate the total distance from the centroidOfSubset to each centroid
    for strokeId in subset:
        centroid = strokeIdToCentroid[strokeId]

        # Generate the Euclidean distances from the subset centroid to each stroke's centroid
        distance += math.sqrt(math.pow(centroidOfSubset[0] - centroid[0], 2) + math.pow(centroidOfSubset[1] - centroid[1], 2))

    return distance

# Returns the list of unique partitions of stroke ids and list of unique partitions of strokes
# restricted by the sum of all distances being less than or equal to the given maxDistance parameter.
# The strokes are Min-Max scaled to a range of [0,maxWidth] and [0,maxHeight] for x and y respectively.
def getUniquePartitionsOfStrokes(strokeIdToStroke, maxDistance, maxStrokesPerSymbol, maxWidth, maxHeight):
    # Get the lists of keys and values
    strokeIds = strokeIdToStroke.keys()

    # Build up centroid positions for the given strokes
    strokeIdToCentroid = calculateCentroidsByStrokeId(strokeIdToStroke, maxWidth, maxHeight)

    # Initialize uniquePartitions
    uniquePartitionsByStrokeIds = []

    # Generate unique partitions by stroke ids (with minimum size of 1 stroke)
    for L in range(1, min(len(strokeIds) + 1, maxStrokesPerSymbol)):
        for subset in itertools.combinations(strokeIds, L):
            #print("subset=", subset)
            uniquePartitionsByStrokeIds.append(subset)

    # Calculate the sum distance from centroid for each subset
    optimizedUniquePartitionsByStrokeIds = []
    for subset in uniquePartitionsByStrokeIds:
        #print("Processing subset=", subset)
        distance = calculateSumOfDistancesToCentroid(subset, strokeIdToCentroid)
        if distance < maxDistance:
            optimizedUniquePartitionsByStrokeIds.append(subset)

    print("Before optimiztion: Number of partitions =", len(uniquePartitionsByStrokeIds))
    print("After optimiztion:  Number of partitions =", len(optimizedUniquePartitionsByStrokeIds))

    return optimizedUniquePartitionsByStrokeIds

# Method to check that the given segmentation is eactly a complete segmentation
#  Meaning, reject segmentations that include the same stroke more than once
#  Meaning, reject segmentations that do not include all strokes
def checkForStrokeCompleteSegmentation(segmentation, strokeIdToStroke, subsetIdToSubset):
    # Build a full set of strokeIds
    strokeIds = list(strokeIdToStroke.keys()).copy()

    # Initialize dictionary with 0 values
    strokeIdToCount = dict()
    for strokeId in strokeIds:
        strokeIdToCount[strokeId] = 0

    # Iterate over all subsetIdentifiers in this segmentation
    for subsetIdentifier in segmentation:
        # Increment our stroke count dictionary
        for strokeId in subsetIdToSubset[subsetIdentifier]:
            strokeIdToCount[strokeId] += 1

    # Check for overlapping strokes (count > 1)
    # Check for missing strokes (count = 0)
    for count in strokeIdToCount.values():
        if count > 1:
            return False
        if count == 0:
            return False

    return True

# Method to use the predicted symbols and probabilities to generate possible segmentation sets
def createSegmentationSets(strokeIdToStroke, uniquePartitionsByStrokeIds, subsetIdToPredictedSymbol, subsetIdToProbability, subsetIdToPredictedSymbolValid, subsetIdToProbabilityValid):
    # First, let's filter down the uniquePartitionsByStrokesIds by eliminating all partitions that result in "Junk"
    subsetIdToSubset = dict()
    for subsetIdentifier in range(0, len(uniquePartitionsByStrokeIds)):
        if subsetIdToPredictedSymbol.get(subsetIdentifier, "junk") != "junk":
            # Found a keeper, this subset is predicted to not be junk
            subsetIdToSubset[subsetIdentifier] = uniquePartitionsByStrokeIds[subsetIdentifier]

    # Display the subsetIdentifiers and predictions that have been kept so far
    for subsetIdentifier in subsetIdToSubset.keys():
        print("Keeping subsetId=", subsetIdentifier, ", symbol=", subsetIdToPredictedSymbol[subsetIdentifier], ", probability=", subsetIdToProbability[subsetIdentifier], ", subset=", subsetIdToSubset[subsetIdentifier])

    # Find the list of strokes that have not participated in the symbol predictions
    remainingStrokeIds = strokeIdToStroke.keys()
    for subset in subsetIdToSubset.values():
        remainingStrokeIds = list(set(remainingStrokeIds) - set(subset))
    print("Remainder strokeIds predicted as junk=", remainingStrokeIds)

    # Re-classify these remainindStrokeIds using the valid symbols classifier
    for remainingStrokeId in remainingStrokeIds:
        subset = []
        subset.append(remainingStrokeId)
        subsetIdToSubset["remainder_" + remainingStrokeId] = subset

    # Construct the combinations of the subsetIds such that maximum strokes are included in each segmentation and there are no duplicates of strokeIds in the segmentation
    # Basically, construct the combinations of subsetIds from 1 to len(subsetIds
    # Generate unique partitions by stroke ids (with minimum size of 1 stroke)
    segmentationSets = []
    for L in range(1, len(subsetIdToSubset.keys()) + 1):
        for segmentation in itertools.combinations(subsetIdToSubset.keys(), L):
            # Check for complete segmentations
            if checkForStrokeCompleteSegmentation(segmentation, strokeIdToStroke, subsetIdToSubset):
                print("segmentation=", segmentation)
                segmentationSets.append(segmentation)

    return segmentationSets

# Segments the given input file, producing an .lg file as output
def segment(fileList, mode):
    # Tuning parameters
    maxDistance = 50            # Relative to min-max scaler of all strokes to maxWidth and maxHeight
    maxStrokesPerSymbol = 5     # No specific support identified for the true upper limit for this
    maxWidth = 100              # Min-Max scaler centroid x coordinates (0, maxWidth)
    maxHeight = 100             # Min-Max scaler centroid y coordinates (0, maxHeight)

    # Load up the models we will use for classification (from Project 1 deliverables)
    rf, encoderModel = loadModels("encoder_both_rf.pkl", "pickle_both_rf.pkl")
    valid_rf, valid_encoderModel = loadModels("encoder_rf.pkl", "pickle_rf.pkl")

    for filename in fileList:
        print("Segmenting ", filename.strip())

        # Setup some lookups that will be used for processing this file
        subsetIdToPredictedSymbol = dict()
        subsetIdToProbability = dict()
        subsetIdToPredictedSymbolValid = dict()
        subsetIdToProbabilityValid = dict()

        # Parse inkml file
        strokeIdToStroke, gt_strokeIdToSymbolId, gt_symbolIdToSymbol = extractStrokes(filename.strip(), mode)

        # Calculates unique partitions of strokes
        uniquePartitionsByStrokeIds = getUniquePartitionsOfStrokes(strokeIdToStroke, maxDistance, maxStrokesPerSymbol, maxWidth, maxHeight)

        # Passes unique partitions of strokes to classifier to get symbols and probabilities, remembering them
        for subsetIdentifier in range(0, len(uniquePartitionsByStrokeIds)):
            # Build up list of strokes raw data for the given subset
            strokes = []
            for strokeId in uniquePartitionsByStrokeIds[subsetIdentifier]:
                strokes.append(strokeIdToStroke[strokeId])

            # Build up the features for this subset of strokes
            uiStack, featureStack, targetStack = getFeaturesForStrokes(subsetIdentifier, strokes)

            # Classify this subset of strokes using the combined classifier
            predictedSymbol, probability = classifyStrokeSubset(featureStack, rf, encoderModel)
            #print("Predictions (combined) for subsetId=", subsetIdentifier, ": predictedSymbol=", predictedSymbol, ", probability=", probability, ", strokes=", uniquePartitionsByStrokeIds[subsetIdentifier])
            subsetIdToPredictedSymbol[subsetIdentifier] = predictedSymbol
            subsetIdToProbability[subsetIdentifier] = probability

            # Classify this subset of strokes using the valid symbols only classifier
            predictedSymbol, probability = classifyStrokeSubset(featureStack, valid_rf, valid_encoderModel)
            #print("Predictions (valid) for subsetId=", subsetIdentifier, ": predictedSymbol=", predictedSymbol, ", probability=", probability, ", strokes=", uniquePartitionsByStrokeIds[subsetIdentifier])
            subsetIdToPredictedSymbolValid[subsetIdentifier] = predictedSymbol
            subsetIdToProbabilityValid[subsetIdentifier] = probability

        # Creates sets of partitions of strokes
        segmentationSets = createSegmentationSets(strokeIdToStroke, uniquePartitionsByStrokeIds, subsetIdToPredictedSymbol, subsetIdToProbability, subsetIdToPredictedSymbolValid, subsetIdToProbabilityValid)

        # Chooses highest probability segmentation
        #segmentation = chooseBestSegmentation(segmentationSets, subsetIdToProbability)

        # Generates .lg output file using symbols and stroke ids
    return

# Main entry point for the program
def main():
    # Initial mode of operation
    mode = "test"

    # Print usage...
    if(len(sys.argv) < 2):
        print("Usage:")
        print("  segmenter.py [<train|test>] <inputFile>")
        print("")
        print("  <train|test>  Optional: Specify 'train' to train the segmenter on a list of files, OR...")
        print("                          Specify 'test' to test the segmenter on a list of files")
        print("  <inputFile>   When NOT train or test, an .inkml file to be segmented to produce ")
        print("                          an .lg file with the same filename prefix.")
        print("                When train or test, a file containing a list of files.")
        print("")
        return

    # Initialize the file list
    fileList = []

    # Extract the first parameter
    firstParameter = sys.argv[1]
    if firstParameter == "train" or firstParameter == "test":
        if firstParameter == "train":
            mode = "train"
        inputFile = sys.argv[2]
        fileList = open(inputFile).readlines()
    else:
        fileList.append(sys.argv[1].strip())

    # Print the fileList and mode
    print("mode=", mode, ", fileList=", fileList)

    # Segment the input files
    segment(fileList, mode)

    return

main()