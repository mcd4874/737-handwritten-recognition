#
# datasplit.py
#
# Authors: Eric Hartman and William Duong
#
# Splits the dataset into train and test splits using a greedy algorithm that
# first seeks to reduce errors in target mean, then seeks to reduce variance
# around the target mean.
#

import xml.etree.ElementTree as ET
import sys
import random
import numpy as np

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

# Return a copy of the given histogram with the symbols for the given
# filename added to its distribution
def addFileToHistogramCopy(histogram, filename):
    # Make a copy of the histogram
    histogramNext = histogram.copy()

    # Get all of the symbols from the inkml file
    symbolList = extractSymbolsFromFile(filename)

    # Add the symbol instances to our new distribution
    for symbol in symbolList:
        # Get current count, 0 if no symbol exists in the distribution yet
        symbolCount = histogramNext.get(symbol, 0)

        # Increment symbolCount
        symbolCount += 1

        # Update the distirbution
        histogramNext[symbol] = symbolCount

    # Return the updated distribution (copy)
    return histogramNext

# Calculates the ratio (or mean) of symbol distributions between train vs (train and test)
def calculateDistance(trainHistogram, testHistogram):
    ratios = []

    # Construct a union of the keys for both histograms
    symbols = list( set(trainHistogram.keys()) | set(testHistogram.keys()))

    # Iterate over all of the known symbols
    for symbol in symbols:
        # Calculate the train[symbol] / (train[symbol] + test[symbol]) ratio
        ratio = float(trainHistogram.get(symbol, 0)) / float((trainHistogram.get(symbol, 0) + testHistogram.get(symbol, 0)))

        # Append the ratio to our list
        ratios.append(ratio)

    # Calculate the averate ratio of our ratios, this is the "distance"
    ratioTotal = 0.0
    for ratio in ratios:
        ratioTotal += ratio
    distance = float(ratioTotal) / float(len(ratios))

    return distance

# Calculates the variance of the ratio of symbol distributions between train vs (train and test)
def calculateVariance(trainHistogram, testHistogram, mean):
    ratios = []

    # Construct a union of the keys for both histograms
    symbols = list( set(trainHistogram.keys()) | set(testHistogram.keys()))

    # Iterate over all of the known symbols
    for symbol in symbols:
        # Calculate the train[symbol] / (train[symbol] + test[symbol]) ratio
        ratio = float(trainHistogram.get(symbol, 0)) / float((trainHistogram.get(symbol, 0) + testHistogram.get(symbol, 0)))

        # Append the ratio to our list
        ratios.append(ratio)

    # Calculate the varaince of our ratios
    SSD = 0.0
    for ratio in ratios:
        SSD += (ratio - mean)*(ratio - mean)
    variance = SSD / float(len(ratios))

    return variance

# Utility method to print a report of the train and test split symbol distributions
def showDistributionStatistics(trainHistogram, testHistogram):
    # Construct a union of the keys for both histograms
    symbols = list( set(trainHistogram.keys()) | set(testHistogram.keys()))

    # Iterate over all of the known symbols
    for symbol in symbols:
        # Calculate the train[symbol] / (train[symbol] + test[symbol]) ratio
        trainRatio = float(trainHistogram.get(symbol, 0)) / float((trainHistogram.get(symbol, 0) + testHistogram.get(symbol, 0)))

        # Calculate the test[symbol] / (train[symbol] + test[symbol]) ratio
        testRatio = float(testHistogram.get(symbol, 0)) / float((trainHistogram.get(symbol, 0) + testHistogram.get(symbol, 0)))

        # Output summary details per symbol
        print("symbol [", symbol, "]: train=", trainRatio, ", test=", testRatio, ", totalCount=", trainHistogram.get(symbol,0) + testHistogram.get(symbol,0))

    return

# Greedy Algorithm to split the train/test files
def greedySplit(inputFile, trainFile, testFile):
    # Histograms for tracking distributions of symbols in train/test splits
    trainHistogram = dict()
    testHistogram = dict()

    # Open output files for writing
    trainOutput = open(trainFile, "w+")
    testOutput = open(testFile, "w+")

    # Extract filenames from input file
    files = open(inputFile).readlines()

    # Randomize the filenames
    random.shuffle(files)

    for file in files:
        filename = file.strip()
        #print("Processing ", filename)

        # Add file to "next" possible histogram copies
        trainHistogramNext = addFileToHistogramCopy(trainHistogram, filename)
        testHistogramNext = addFileToHistogramCopy(testHistogram, filename)

        # Choose split based on minimizing variance...
        trainNextVariance = calculateVariance(trainHistogramNext, testHistogram, 0.70)
        testNextVariance = calculateVariance(trainHistogram, testHistogramNext, 0.70)

        # Greedily choose winner as the one leading to lowest variance
        if trainNextVariance < testNextVariance:
            # Add to Train
            trainHistogram = trainHistogramNext
            trainOutput.write("" + filename + "\n")
            print("Variance = ", trainNextVariance)
        else:
            # Add to Test
            testHistogram = testHistogramNext
            testOutput.write("" + filename + "\n")
            print("Variance = ", testNextVariance)

    # Close the output files
    testOutput.close()
    trainOutput.close()

    # Output the symbol distribution statistics
    showDistributionStatistics(trainHistogram, testHistogram)

    return

# Main entry point for the program
def main():
    # Print usage...
    if(len(sys.argv) < 4):
        print("Usage:")
        print("  datasplit.py <inputFile> <trainFile> <testFile>")
        print("")
        print("  <inputFile> contains a list of .inkml filenames with full paths")
        print("  <trainFile> will contain a list of files for the train split")
        print("  <testFile> will contain a list of files for the test split")
        print("")
        return

    # Extract the input file
    inputFile = sys.argv[1]

    # Extract the train file
    trainFile = sys.argv[2]

    # Extract the test file
    testFile = sys.argv[3]

    print("Input:", inputFile, ", Train:", trainFile, ", Test:", testFile)

    # Perform a greedy split of train/test
    greedySplit(inputFile, trainFile, testFile)

    return

main()