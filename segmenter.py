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
# The strokes are Min-Max scaled to a range of [0,maxWidth] and [0,maxHeight] for x and y respectively,
# but this is done preserving the aspect ratio.  To preserve the aspect ratio, an alternate of
# either maxWidth or maxHeight will be calculated and used for this Min-Max scaling.
# The "shortest" true dimension is scaled to 100, the longest true dimension is scaled to a ratio
# of the shortest dimension * 100
def calculateCentroidsByStrokeIdPreservingAspectRatio(strokeIdToStroke, maxWidth, maxHeight):
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

    # Use the peaks and troughs to preserve aspect ratio
    trueHeight = abs(peakY - troughY)
    trueWidth = abs(peakX - troughX)
    scaleWidth = maxWidth
    scaleHeight = maxHeight
    if trueHeight > trueWidth:
        # Height is the primary dimension, scale Height accordingly to multiple of maxHeight
        scaleHeight = float(float(trueHeight) / float(trueWidth)) * maxHeight
    else:
        # Width is the primary dimension, scale Width accordingly to multiple of maxWidth
        scaleWidth = float(float(trueWidth) / float(trueHeight)) * maxWidth

    # Min-Max scale the centroid positions according to the global peaks and troughs
    scalerX = MinMaxScaler(feature_range=(0, scaleWidth), copy=False)
    scalerY = MinMaxScaler(feature_range=(0, scaleHeight), copy=False)

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


# Helper method to calculate the centroid of a subset of strokes
def calculateCentroidOfSubset(subset, strokeIdToCentroid):
    # Calculate the centroid of the subset
    x = 0.0
    y = 0.0
    for strokeId in subset:
        strokeCentroid = strokeIdToCentroid[strokeId]
        x += strokeCentroid[0]
        y += strokeCentroid[1]

    centroidOfSubset = [float(x) / float(len(subset)), float(y) / float(len(subset))]
    return centroidOfSubset

# Returns the total distance from the centroid of the subset to every member of the subset
# Something like a "star distance"
def calculateSumOfDistancesToCentroid(subset, strokeIdToCentroid):
    distance = 0.0

    # Calculate the centroid of the subset
    centroidOfSubset = calculateCentroidOfSubset(subset, strokeIdToCentroid)

    # Calculate the total distance from the centroidOfSubset to each centroid
    for strokeId in subset:
        centroid = strokeIdToCentroid[strokeId]

        # Generate the Euclidean distances from the subset centroid to each stroke's centroid
        distance += math.sqrt(math.pow(centroidOfSubset[0] - centroid[0], 2) + math.pow(centroidOfSubset[1] - centroid[1], 2))

    return distance

# Calculates the maximum distance of any stroke centroid to the subset's centroid
def calculateMaxDistanceToCentroid(subset, strokeIdToCentroid):
    maxDistance = 0.0

    # Calculate the centroid of the subset
    centroidOfSubset = calculateCentroidOfSubset(subset, strokeIdToCentroid)

    # Calculate the distance from the centroidOfSubset to furthest centroid
    for strokeId in subset:
        centroid = strokeIdToCentroid[strokeId]

        # Generate the Euclidean distances from the subset centroid to each stroke's centroid
        distance = math.sqrt(math.pow(centroidOfSubset[0] - centroid[0], 2) + math.pow(centroidOfSubset[1] - centroid[1], 2))

        # Check if this is max so far
        if distance > maxDistance:
            maxDistance = distance

    return maxDistance

# Calculates whether all of the stroke centroids are within the limitDistance to subset's centroid
def withinLimitDistanceToCentroid(subset, strokeIdToCentroid, limitDistance):
    # Calculate the centroid of the subset
    centroidOfSubset = calculateCentroidOfSubset(subset, strokeIdToCentroid)

    # Calculate the distance from the centroidOfSubset to furthest centroid
    for strokeId in subset:
        centroid = strokeIdToCentroid[strokeId]

        # Generate the Euclidean distances from the subset centroid to each stroke's centroid
        distance = math.sqrt(math.pow(centroidOfSubset[0] - centroid[0], 2) + math.pow(centroidOfSubset[1] - centroid[1], 2))

        # Check if limit has been exceeded
        if distance > limitDistance:
            return False

    return True

# Returns the list of unique partitions of stroke ids and list of unique partitions of strokes
# restricted by the sum of all distances being less than or equal to the given maxDistance parameter.
# The strokes are Min-Max scaled to a range of [0,maxWidth] and [0,maxHeight] for x and y respectively.
def getUniquePartitionsOfStrokes(strokeCountToSetSizeToCombinations, strokeIdToStroke, maxDistance, maxStrokesPerSymbol, maxWidth, maxHeight):
    # Get the lists of keys and values
    strokeIds = strokeIdToStroke.keys()

    # Build up centroid positions for the given strokes
    #strokeIdToCentroid = calculateCentroidsByStrokeId(strokeIdToStroke, maxWidth, maxHeight)
    strokeIdToCentroid = calculateCentroidsByStrokeIdPreservingAspectRatio(strokeIdToStroke, maxWidth, maxHeight)

    # Initialize uniquePartitions
    uniquePartitionsByStrokeIds = []

    # Initialize single stroke mappings for use later
    strokeIdToSubsetId = dict()
    strokeIdToSubset = dict()

    # Generate unique partitions by stroke ids (with minimum size of 1 stroke)
    for L in range(1, min(len(strokeIds) + 1, maxStrokesPerSymbol + 1)):
        # See if we already have cached combinations to use...
        strokeCount = len(strokeIds)

        #print("strokeCountToSetSizeToCombinations count=", len(strokeCountToSetSizeToCombinations.keys()))

        # Use caching to increase performance
        strokeCountToSetSize = strokeCountToSetSizeToCombinations.get(strokeCount, None)
        if strokeCountToSetSize == None:
            strokeCountToSetSize = dict()
            strokeCountToSetSizeToCombinations[strokeCount] = strokeCountToSetSize

        # Use caching to increase performance
        setSizeToCombinations = strokeCountToSetSize.get(L, dict())
        strokeCountToSetSize[L] = setSizeToCombinations

        # Use caching to increase performance
        combinations = setSizeToCombinations.get(L, None)
        if combinations == None:
            combinations = list(itertools.combinations(strokeIds, L))
            setSizeToCombinations[L] = combinations
            print("Caching combinations for setSize [", L, "], combinations=", len(combinations))
        else:
            print("Re-using cached combinations for setSize [", L, "], combinations=", len(combinations))
        #combinations = setSizeToCombinations.get(L, itertools.combinations(strokeIds, L))

        #for subset in itertools.combinations(strokeIds, L):
        for subset in combinations:
            #print("subset=", subset)
            uniquePartitionsByStrokeIds.append(subset)

            # Remember the single-stroke mappings to subsetIdentifiers and subsets
            # These will be used in createSegmentationSets
            if L == 1:
                strokeIdToSubsetId[subset[0]] = len(uniquePartitionsByStrokeIds) - 1
                strokeIdToSubset[subset[0]] = subset

    print("Before proximity optimization: Number of partitions =", len(uniquePartitionsByStrokeIds))

    # Calculate the sum distance from centroid for each subset
    optimizedUniquePartitionsByStrokeIds = []
    for subset in uniquePartitionsByStrokeIds:
        #print("Processing subset=", subset)
        #distance = calculateSumOfDistancesToCentroid(subset, strokeIdToCentroid)

        # TODO- move distance into calculateMaxDistance
        #distance = calculateMaxDistanceToCentroid(subset, strokeIdToCentroid)
        #if distance < maxDistance:
        if withinLimitDistanceToCentroid(subset, strokeIdToCentroid, maxDistance) == True:
            optimizedUniquePartitionsByStrokeIds.append(subset)

    print("After proximity optimization:  Number of partitions =", len(optimizedUniquePartitionsByStrokeIds))

    return optimizedUniquePartitionsByStrokeIds, strokeIdToSubsetId, strokeIdToSubset

# Method to check that the given segmentation is eactly a complete segmentation
#  Meaning, reject segmentations that include the same stroke more than once
#  Meaning, reject segmentations that do not include all strokes
def checkForStrokeCompleteSegmentation(segmentation, strokeIdToStroke, subsetIdToSubset, strokeIdToSubsetId, subsetIdToPredictedSymbolValid, subsetIdToPredictedSymbol):
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
    for strokeId in strokeIdToCount.keys():
        count = strokeIdToCount[strokeId]
        if count > 1:
            return False
        if count == 0:
            # Add the single stroke subsetIdentifier to our segment, so we eliminate all Absences
            strokeSubset = []
            strokeSubset.append(strokeId)
            subsetIdentifier = int(strokeId)  # Special case only works for single-stroke subsets
            subsetIdToSubset[subsetIdentifier] = strokeSubset
            segmentation.append(subsetIdentifier)
            print("Adding absent stroke's subset to segmentation: strokeId=", strokeId, ", subsetId=", strokeIdToSubsetId[strokeId])

    return True

# Recursive method
def generateSegmentationSetsRecurse(segmentationSets, subsetIdToSubset, subsetIdentifier, inprogressSegment, inprogressStrokeIdToCount, strokeIdToStroke):
    # Process the subset identifier...

    # Iterate over the strokes in the subset and add to inprogressStrokeIdToCount
    for strokeId in subsetIdToSubset[subsetIdentifier]:
        count = inprogressStrokeIdToCount.get(strokeId, 0)
        count += 1

        # Determine if we've violated the stroke overlap rule
        if count > 1:
            #print("Violated overlap rule!")
            return
        else:
            inprogressStrokeIdToCount[strokeId] = count

    # Let's add the subset identifier to the inprogressSegment
    inprogressSegment.append(subsetIdentifier)

    # Determine if we've completed the segment by incorporating ALL strokes exactly once
    strokesUsed = 0
    for count in inprogressStrokeIdToCount.values():
        strokesUsed += count
    if strokesUsed == len(strokeIdToStroke.keys()):
        # We're done!
        #print("Found candidate segmentation: ", inprogressSegment)
        segmentationSets.append(inprogressSegment)
        return

    # Identify remaining subsetIdentifiers...
    remainingSubsetIdentifiers = list(set(subsetIdToSubset.keys()) - set(inprogressSegment))

    # Iterate over remaining subsetIdentifiers...
    for remainingSubsetIdentifier in remainingSubsetIdentifiers:
        # Recurse to next level...
        inprogressSegmentCopy = inprogressSegment.copy()
        inprogressStrokeIdToCountCopy = inprogressStrokeIdToCount.copy()
        generateSegmentationSetsRecurse(segmentationSets, subsetIdToSubset, remainingSubsetIdentifier, inprogressSegmentCopy, inprogressStrokeIdToCountCopy, strokeIdToStroke)

    return

# Recursively construct the segmentation sets without overlapping stroke ids
def generateSegmentationSets(strokeIdToStroke, subsetIdToSubset):
    segmentationSets = []

    # Iterate over subsets
    for subsetIdentifier in subsetIdToSubset.keys():
        inprogressSegment = []
        inprogressStrokeIdToCount = dict()
        generateSegmentationSetsRecurse(segmentationSets, subsetIdToSubset, subsetIdentifier, inprogressSegment, inprogressStrokeIdToCount, strokeIdToStroke)

    return segmentationSets

# Method to use the predicted symbols and probabilities to generate possible segmentation sets
def createSegmentationSets(minProbabilitySingle, minProbabilityMultiple, maxSymbols, strokeIdToStroke, uniquePartitionsByStrokeIds, subsetIdToPredictedSymbol, subsetIdToProbability, subsetIdToPredictedSymbolValid,
                           subsetIdToProbabilityValid, strokeIdToSubsetId, strokeIdToSubset):
    # First, let's filter down the uniquePartitionsByStrokesIds by eliminating all partitions that result in "Junk"
    subsetIdToSubset = dict()
    for subsetIdentifier in range(0, len(uniquePartitionsByStrokeIds)):
        strokeCount = len(uniquePartitionsByStrokeIds[subsetIdentifier])
        probabilityValid = subsetIdToProbabilityValid[subsetIdentifier]
        probabilityCombined = subsetIdToProbability[subsetIdentifier]

        predictedSymbolValid = subsetIdToPredictedSymbolValid.get(subsetIdentifier, "junk")
        predictedSymbol = subsetIdToPredictedSymbol.get(subsetIdentifier, "junk")
        if predictedSymbol != "junk":
            # Special case for "+" symbol, this requires 2 or fewer symbols.  When considering many strokes, it
            # is easy for multiple strokes to be classified as a "+", for example: 1 / 1 looks like "+"
            if predictedSymbol == "+" and len(uniquePartitionsByStrokeIds[subsetIdentifier]) > 2:
                # False "+"
                continue

            # Skip any subsets that don't meet our probability theshold
            if strokeCount > 1 and probabilityCombined < minProbabilityMultiple:
                print("multi-stroke subsetId=", subsetIdentifier, " has too low probability: ", probabilityCombined, " and ", probabilityValid)
                continue
            elif strokeCount == 1 and probabilityCombined < minProbabilitySingle:
                print("singlestroke subsetId=", subsetIdentifier, " has too low probability: ", probabilityCombined, " and ", probabilityValid)
                continue

            # Found a keeper, this subset is predicted to not be junk
            subsetIdToSubset[subsetIdentifier] = uniquePartitionsByStrokeIds[subsetIdentifier]
        else:
            print("Not keeping subsetid=", subsetIdentifier, ", symbol=", predictedSymbol, ", probability=", subsetIdToProbability[subsetIdentifier], ", subset=", uniquePartitionsByStrokeIds[subsetIdentifier], ", vsymbol=", predictedSymbolValid, ", vprobability=", probabilityValid)

    # Display the subsetIdentifiers and predictions that have been kept so far
    for subsetIdentifier in subsetIdToSubset.keys():
        print("Keeping subsetId=", subsetIdentifier, ", symbol=", subsetIdToPredictedSymbol[subsetIdentifier], ", probability=", subsetIdToProbability[subsetIdentifier], ", subset=", subsetIdToSubset[subsetIdentifier])

    # Find the list of strokes that have not participated in the symbol predictions
    remainingStrokeIds = strokeIdToStroke.keys()
    for subset in subsetIdToSubset.values():
        remainingStrokeIds = list(set(remainingStrokeIds) - set(subset))
    print("Remainder strokeIds predicted as junk=", remainingStrokeIds)

    # Re-classify these remainindStrokeIds using the valid symbols classifier
    remainingStrokeSubsetIds = []
    for remainingStrokeId in remainingStrokeIds:
        subset = strokeIdToSubset[remainingStrokeId]
        subsetIdentifier = strokeIdToSubsetId[remainingStrokeId]
        #subset.append(remainingStrokeId)
        #subsetIdToSubset["remainder_" + remainingStrokeId] = subset
        subsetIdToSubset[subsetIdentifier] = subset
        remainingStrokeSubsetIds.append(subsetIdentifier)

    # Find the subsets containing only "1-count" strokes, these subsets MUST be in every segmentation
    #prefix = remainingStrokeSubsetIds
    prefix = []
    strokeIdToCount = dict()
    for subsetIdentifier in subsetIdToSubset.keys():
        subset = subsetIdToSubset[subsetIdentifier]
        for strokeId in subset:
            count = strokeIdToCount.get(strokeId, 0)
            count += 1
            strokeIdToCount[strokeId] = count
    for subsetIdentifier in subsetIdToSubset.keys():
        # Check every stroke to see if this subset contains only "1-count" strokes
        count = 0
        subset = subsetIdToSubset[subsetIdentifier]
        for strokeId in subset:
            count += strokeIdToCount[strokeId]
        if count == len(subset):
            # Found subset containing all "1-count" strokes, add to our prefix
            prefix.append(subsetIdentifier)

    # Construct the combinations of the subsetIds such that maximum strokes are included in each segmentation and there are no duplicates of strokeIds in the segmentation
    # Basically, construct the combinations of subsetIds from 1 to len(subsetIds
    # Generate unique partitions by stroke ids (with minimum size of 1 stroke)
    #segmentationSets = generateSegmentationSets(strokeIdToStroke, subsetIdToSubset)
    segmentationSets = []

    # Optimization
    #   Every segmentation MUST include all of the "remaining" strokes (as subsets), so these subsets
    #   should prefix every segmentation possibility, and we should only permute over
    #   the 'allSubsets' - 'remainder' as concatenations
    #   Also... every segmentation must include all of the subsets containing only strokeIds
    #   that are not included in any other subset.
    allSubsetsRemainder = list(set(subsetIdToSubset.keys()) - set(prefix))

    # Optimization
    #   Eliminate all subsets that are below our minProbability threshold
    #   This will reduce the problem space by eliminating unlikely choices
    highProbabilitySubsetsRemainder = []
    for subsetIdentifier in allSubsetsRemainder:
        strokeCount = len(subsetIdToSubset[subsetIdentifier])
        combinedSymbol = subsetIdToPredictedSymbol[subsetIdentifier]
        combinedProbability = subsetIdToProbability[subsetIdentifier]
        validProbability = subsetIdToProbabilityValid[subsetIdentifier]

        # Apply the correct probability threshold depending on stroke composition
        minProbability = minProbabilityMultiple
        if strokeCount == 1:
            minProbability = minProbabilitySingle

        if combinedSymbol == "junk":
            if validProbability > minProbability:
                # Meets our minimum probability threshold
                highProbabilitySubsetsRemainder.append(subsetIdentifier)
            else:
                print("dropping low probability: subsetId=", subsetIdentifier, ", probability=", validProbability, ", subset=", subsetIdToSubset[subsetIdentifier])
        else:
            if combinedProbability > minProbability:
                # Meets our minimum probability threshold
                highProbabilitySubsetsRemainder.append(subsetIdentifier)
            else:
                print("dropping low probability: subsetId=", subsetIdentifier, ", probability=", combinedProbability, ", subset=", subsetIdToSubset[subsetIdentifier])
    print("dropped low probability subset count =", len(allSubsetsRemainder) - len(highProbabilitySubsetsRemainder))

    # If everything is in the prefix, then the prefix is our segmentation
    if len(highProbabilitySubsetsRemainder) == 0:
        segmentationSets.append(prefix)

    print("prefix=", prefix)

    # Optimization
    #   Limit the maxSymbols to the minimum of maxSymbols or # strokes ( less size of prefix
    # <prefix.......> <highProbabilitySubsetsRemainder....>
    # ----------------------|  <- strokes
    #                |------|  <- strokes - prefix is the combinatoric piece
    #maxSymbols = min(len(strokeIdToStroke.keys()) - len(prefix), maxSymbols - len(prefix))
    maxSymbols = len(strokeIdToStroke.keys()) - len(prefix)

    for L in range(1, min(len(highProbabilitySubsetsRemainder) + 1, maxSymbols + 1)):
        print("About to get combinations for L=", L, ", totalSubsets=", len(highProbabilitySubsetsRemainder), ", prefix=", len(prefix), ", maxSymbols=", maxSymbols,", totalStrokes=", len(strokeIdToStroke.keys()))
        for segmentation in itertools.combinations(highProbabilitySubsetsRemainder, L):
            # Check for complete segmentations
            segmentation = list(prefix) + list(segmentation)
            #print("candidateSegmentation=", segmentation)
            if checkForStrokeCompleteSegmentation(segmentation, strokeIdToStroke, subsetIdToSubset, strokeIdToSubsetId, subsetIdToPredictedSymbolValid, subsetIdToPredictedSymbol):
                #print("segmentation=", segmentation)
                segmentationSets.append(segmentation)

    return segmentationSets, subsetIdToSubset

# Method to select the best segmentation among those that are available
def chooseBestSegmentation(segmentationSets, subsetIdToPredictedSymbol, subsetIdToProbability, subsetIdToProbabilityValid, subsetIdToSubset, multiStrokeBonusProbability):
    # Return the best segmentation
    maxScore = -99999.0
    bestSegmentation = None

    # Iterate over the segmentation sets
    for segmentation in segmentationSets:
        # segmentation is a list of subsetIdentifiers
        score = 0.0
        for subsetIdentifier in segmentation:
            # Junk symbol predictions are replaced with Valid symbol predictions
            #print("DEBUG: subsetIdentifier=", subsetIdentifier, ", in=", subsetIdentifier in subsetIdToPredictedSymbol.keys(), ", type=", type(subsetIdentifier))
            if subsetIdToPredictedSymbol[subsetIdentifier] == "junk":
                if len(subsetIdToSubset[subsetIdentifier]) > 1:
                    # Multiple strokes!
                    score += (subsetIdToProbabilityValid[subsetIdentifier] + multiStrokeBonusProbability) * len(subsetIdToSubset[subsetIdentifier])
                else:
                    # One stroke
                    score += subsetIdToProbabilityValid[subsetIdentifier]
            else:
                if len(subsetIdToSubset[subsetIdentifier]) > 1:
                    # Multiple strokes!
                    score += (subsetIdToProbability[subsetIdentifier] + multiStrokeBonusProbability) * len(subsetIdToSubset[subsetIdentifier])
                else:
                    # One stroke
                    score += subsetIdToProbability[subsetIdentifier]

        # Check new max?
        if score > maxScore:
            maxScore = score
            bestSegmentation = segmentation

        # Report out score...
        print("Segmentation: ", segmentation, ", score=", score)

    # TO-DO: For now, just return the first segmentation
    return bestSegmentation

# Method to create a label graph (.lg) output file for the given segmentation
def generateLabelGraphOutputFile(filename, segmentation, subsetIdToPredictedSymbol, subsetIdToPredictedSymbolValid, subsetIdToSubset):
    # Calculate filename prefix
    prefix = filename.split(".inkml")[0]

    # Output filename
    outputFilename = prefix + ".lg"
    print("outputFilename=", outputFilename)

    outputFile = open(outputFilename, "w+")

    # Use dictionary as symbol counter
    symbolToCount = dict()

    # Iterate over the subsets in the segmentation
    for subsetIdentifier in segmentation:
        # Lookup the predicted symbol...
        symbol = subsetIdToPredictedSymbol[subsetIdentifier]
        if symbol == "junk":
            symbol = subsetIdToPredictedSymbolValid[subsetIdentifier]

        # Construct a unique symbol identifier with pattern <symbol>_<counter>
        count = symbolToCount.get(symbol, 0)
        count += 1
        symbolToCount[symbol] = count
        symbolUI = "" + symbol + "_" + str(count)

        # Write the output...
        outputFile.write("O, " + symbolUI + ", " + symbol + ", 1.0")
        for strokeId in subsetIdToSubset[subsetIdentifier]:
            outputFile.write(", " + strokeId)
        outputFile.write("\n")

    outputFile.close()

    return

# Helper method to perform segmentation using the provided tuning parameters
def segment_helper(fileList, mode, maxDistance, maxStrokesPerSymbol, maxWidth, maxHeight, multiStrokeBonusProbability, maxSymbols, minProbabilitySingle, minProbabilityMultiple):

    # Load up the models we will use for classification (from Project 1 deliverables)
    rf, encoderModel = loadModels("encoder_both_rf.pkl", "pickle_both_rf.pkl")
    valid_rf, valid_encoderModel = loadModels("encoder_rf.pkl", "pickle_rf.pkl")

    # Speed enhancement: Don't recompute combination lists
    strokeCountToSetSizeToCombinations = dict()

    for filename in fileList:

        # Setup some lookups that will be used for processing this file
        subsetIdToPredictedSymbol = dict()
        subsetIdToProbability = dict()
        subsetIdToPredictedSymbolValid = dict()
        subsetIdToProbabilityValid = dict()

        # Parse inkml file
        strokeIdToStroke, gt_strokeIdToSymbolId, gt_symbolIdToSymbol = extractStrokes(filename.strip(), mode)
        print("Segmenting ", filename.strip(), ", # Strokes =", len(strokeIdToStroke.keys()))

        # Skip any file that has zero strokes
        if len(strokeIdToStroke.keys()) == 0:
            print("File has no strokes, nothing to do, so going to skip it.")
            continue

        # Calculates unique partitions of strokes
        uniquePartitionsByStrokeIds, strokeIdToSubsetId, strokeIdToSubset = getUniquePartitionsOfStrokes(strokeCountToSetSizeToCombinations, strokeIdToStroke, maxDistance, maxStrokesPerSymbol, maxWidth, maxHeight)

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
            #print("PredictedSymbol[", subsetIdentifier, "] = ", predictedSymbol)

            # Classify this subset of strokes using the valid symbols only classifier
            predictedSymbol, probability = classifyStrokeSubset(featureStack, valid_rf, valid_encoderModel)
            #print("Predictions (valid) for subsetId=", subsetIdentifier, ": predictedSymbol=", predictedSymbol, ", probability=", probability, ", strokes=", uniquePartitionsByStrokeIds[subsetIdentifier])
            subsetIdToPredictedSymbolValid[subsetIdentifier] = predictedSymbol
            subsetIdToProbabilityValid[subsetIdentifier] = probability

        # Creates sets of partitions of strokes
        segmentationSets, subsetIdToSubset = createSegmentationSets(minProbabilitySingle, minProbabilityMultiple, maxSymbols, strokeIdToStroke, uniquePartitionsByStrokeIds, subsetIdToPredictedSymbol, subsetIdToProbability, subsetIdToPredictedSymbolValid, subsetIdToProbabilityValid,
                                                  strokeIdToSubsetId, strokeIdToSubset)

        # Chooses highest probability segmentation
        segmentation = chooseBestSegmentation(segmentationSets, subsetIdToPredictedSymbol, subsetIdToProbability, subsetIdToProbabilityValid, subsetIdToSubset, multiStrokeBonusProbability)

        # Generates .lg output file using symbols and stroke ids
        generateLabelGraphOutputFile(filename, segmentation, subsetIdToPredictedSymbol, subsetIdToPredictedSymbolValid, subsetIdToSubset)

    return

# Segments the given input file, producing an .lg file as output using a baseline segmentation approach
def real_segment(fileList, mode):
    # Tuning parameters
    maxDistance = 20                    # Relative to min-max scaler of all strokes to maxWidth and maxHeight
    maxStrokesPerSymbol = 5             # No specific support identified for the true upper limit for this
    maxWidth = 100                      # Min-Max scaler centroid x coordinates (0, maxWidth)
    maxHeight = 100                     # Min-Max scaler centroid y coordinates (0, maxHeight)
    multiStrokeBonusProbability = 0.75  # Bonus probability added to real probability before applying length multiplier
    maxSymbols = 30                     # Maximum number of symbols to support for combinatorics
    minProbabilitySingle = 0.50 #0.60   # Threshold for eliminating subsets (containing 1 stroke) with weak probabilities
    minProbabilityMultiple = 0.40 #0.50 # Threshold for eliminating subsets (containing multiple strokes) with weak probabilities
    #maxSymbolToStrokeRatio = 0.75      # Threshold for limiting max symbols to no more than this ratio against Stroke count

    segment_helper(fileList, mode,
                   maxDistance=maxDistance,
                   maxStrokesPerSymbol=maxStrokesPerSymbol,
                   maxWidth=maxWidth,
                   maxHeight=maxHeight,
                   multiStrokeBonusProbability=multiStrokeBonusProbability,
                   maxSymbols=maxSymbols,
                   minProbabilitySingle=minProbabilitySingle,
                   minProbabilityMultiple=minProbabilityMultiple)
    return

# Segments the given input file, producing an .lg file as output using a baseline segmentation approach
def baseline_segment(fileList, mode):
    # Tuning parameters
    maxDistance = 5                     # Relative to min-max scaler of all strokes to maxWidth and maxHeight
    maxStrokesPerSymbol = 2             # No specific support identified for the true upper limit for this
    maxWidth = 100                      # Min-Max scaler centroid x coordinates (0, maxWidth)
    maxHeight = 100                     # Min-Max scaler centroid y coordinates (0, maxHeight)
    multiStrokeBonusProbability = 0.0   # Bonus probability added to real probability before applying length multiplier
    maxSymbols = 30                     # Maximum number of symbols to support for combinatorics
    minProbabilitySingle = 0.00         # Threshold for eliminating subsets (containing 1 stroke) with weak probabilities
    minProbabilityMultiple = 0.00       # Threshold for eliminating subsets (containing multiple strokes) with weak probabilities

    segment_helper(fileList, mode,
                   maxDistance=maxDistance,
                   maxStrokesPerSymbol=maxStrokesPerSymbol,
                   maxWidth=maxWidth,
                   maxHeight=maxHeight,
                   multiStrokeBonusProbability=multiStrokeBonusProbability,
                   maxSymbols=maxSymbols,
                   minProbabilitySingle=minProbabilitySingle,
                   minProbabilityMultiple=minProbabilityMultiple)

    return

# Main entry point for the program
def main():
    # Initial mode of operation
    mode = "real"

    # Print usage...
    if(len(sys.argv) < 2):
        print("Usage:")
        print("  segmenter.py <real|baseline> <inputFile>")
        print("")
        print("  <real|baseline>  Optional: Specify 'real' to use sophisticated segmenter on a list of files, OR...")
        print("                             Specify 'baseline' to use baseline segmenter on a list of files")
        print("  <inputFile>      Either an .inkml file to be segmented or a file containing a list of .inkml files ")
        print("")
        return

    # Initialize the file list
    fileList = []

    # Extract the first parameter
    firstParameter = sys.argv[1]
    if firstParameter == "real" or firstParameter == "baseline":
        if firstParameter == "real":
            mode = "real"
        else:
            mode = "baseline"

        # Determine if we are processing an individual file or file list
        if ".inkml" in sys.argv[2]:
            fileList.append(sys.argv[2].strip())
        else:
            inputFile = sys.argv[2]
            fileList = open(inputFile).readlines()
    else:
        return

    # Print the fileList and mode
    print("mode=", mode, ", fileList=", fileList)

    # Segment the input files
    if mode == "real":
        real_segment(fileList, mode)
    if mode == "baseline":
        print("Using baseline segmenter.")
        baseline_segment(fileList, mode)

    return

main()