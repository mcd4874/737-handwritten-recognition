from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import xml.etree.ElementTree as ET
import os
import cv2
import csv
import sys

# Extract the list of symbols from the given inkml file
def extractSymbolsFromFile(filename):
    # Initialize our list of symbols
    symbols = []

    # Define our ParseError
    ParseError = ET.ParseError;

    try:
        # Parse inkml and get root
        tree = ET.parse(filename)

        # extract annotation type
        for node in tree.findall('.//{http://www.w3.org/2003/InkML}traceGroup/{http://www.w3.org/2003/InkML}traceGroup/{http://www.w3.org/2003/InkML}annotation'):
            symbols.append(node.text.strip())

        #print("File: ", filename, ", symbols: ", symbols)

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

# Calculates the ratio of symbol distributions between train vs (train and test)
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
    distance = float(ratioTotal) / float(len(ratios));

    return distance

# Greedy Algorithm to split the train/test files
def greedySplit(inputFile, trainFile, testFile):
    # Histograms for tracking distributions of symbols in train/test splits
    trainHistogram = dict()
    testHistogram = dict()

    # Open output files for writing
    trainOutput = open(trainFile, "w+")
    testOutput = open(testFile, "w+")

    files = open(inputFile)
    for file in files:
        filename = file.strip()
        print("Processing ", filename)

        # Add file to "next" possible histogram copies
        trainHistogramNext = addFileToHistogramCopy(trainHistogram, filename)
        testHistogramNext = addFileToHistogramCopy(testHistogram, filename)

        # Calculate the distances from current to possible future states
        # as a ratio of future state "train" / "train + test"
        trainNextDistance = calculateDistance(trainHistogramNext, testHistogram)
        testNextDistance = calculateDistance(trainHistogram, testHistogramNext)

        # Calculate the train and test errors (70% is target ratio for train vs total)
        trainNextError = abs(trainNextDistance - 0.70)
        testNextError = abs(testNextDistance - 0.70)

        # Greedily choose winner as the one leading to lowest error
        if trainNextError < testNextError:
            # Add to Train
            trainHistogram = trainHistogramNext
            trainOutput.write("" + filename + "\n")
            print("Distance = ", trainNextDistance)
        else:
            # Add to Test
            testHistogram = testHistogramNext
            testOutput.write("" + filename + "\n")
            print("Distance = ", testNextDistance)

    # Close the output files
    testOutput.close()
    trainOutput.close()

    # Close the input file
    files.close()

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