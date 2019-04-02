#
# The following shell commands will generate the frequencies for the Real Symbols dataset
#   cat iso_GT.txt | awk - F',' {' print $2 '} | sort | uniq - c | sort > classFrequencies.txt
#
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
import math
from generateFeatureStack import cacheUItoFilename

def generateTrainTestSplit(sourceFile, trainFile, testFile, uiToFilename, testFileForEval):
    # Extract the path prefix
    elements = sourceFile.split('/')
    prefix = "./" + elements[1] + "/"

    # Load the dictionary file
    data = np.genfromtxt(sourceFile, delimiter=',', dtype=str)

    # Collect up the class labels
    classes, counts = np.unique(data[:, 1], return_counts=True)
    print("Classes=", classes)
    print("Counts=", counts)

    # Open the output files for writing
    fTest = open(testFile, "w+")
    fTrain = open(trainFile, "w+")
    fEval = open(testFileForEval, "w+")

    # 70% Train, 30% Test
    for i in range(0, len(classes)):
        # Extract the symbol samples that match the class
        symbolSamples = data[data[:, 1] == classes[i]]

        # Determine Test count
        testCount = int(len(symbolSamples) * 0.30)

        # Determine Train count
        trainCount = len(symbolSamples) - testCount

        print("testCount=", testCount, ", trainCount=", trainCount, ", totalCount=",
              len(symbolSamples))

        # Write Test samples (0, testCount)
        for j in range(0, testCount):
            #fTest.write('' + symbolSamples[j][0] + ',' + symbolSamples[j][1] + '\n')
            print("key=", symbolSamples[j][0], ", filename=", uiToFilename[symbolSamples[j][0]])
            fTest.write('' + uiToFilename[symbolSamples[j][0]] + ',' + symbolSamples[j][1] + '\n')

            # Write out the "evalSymIsole.py" compatible test output file
            fEval.write('' + symbolSamples[j][0] + ',' + symbolSamples[j][1] + '\n')

        # Write Train samples (testCount, totalCount)
        for j in range(testCount, len(symbolSamples)):
            fTrain.write('' + symbolSamples[j][0] + ',' + symbolSamples[j][1] + '\n')

    # Close the output files
    fTest.close()
    fTrain.close()
    fEval.close()

    return

def generateBalancedClasses(sourceFile, outputFile):
    # Load the dictionary file
    data = np.genfromtxt(sourceFile, delimiter=',', dtype=str)
    dataResampled = np.copy(data)

    # Collect up the class labels
    classes, counts = np.unique(data[:, 1], return_counts=True)
    print("Classes=", classes)
    print("Counts=", counts)

    # Find the maximum count
    maxFrequency = np.max(counts)
    print("MaxFrequency=", maxFrequency)

    # Upsampling: Rebalance all minority classes to maxFrequency
    for i in range(0, len(classes)):
        # Extract the symbol samples that match the class
        symbolSamples = data[data[:, 1] == classes[i]]

        # Calculate the number of resamples needed
        resamples = maxFrequency - counts[i]

        # Resample the needed quantity
        symbolResamples = resample(symbolSamples,
                                   replace=True,
                                   n_samples=resamples,
                                   random_state=8675309)

        # Concatenate resamples to dataResampled
        dataResampled = np.concatenate((dataResampled, symbolResamples), axis=0)

        print("Symbol[", i, "]: ", classes[i], ", real count=", len(symbolSamples), ", resamples=",
              resamples, ", upsample total=", len(symbolResamples) + len(symbolSamples))

    # Shuffle the resampled data
    np.random.shuffle(dataResampled)

    # Write the output file
    f = open(outputFile, "w+")
    for i in range(len(dataResampled)):
        f.write('' + dataResampled[i][0] + ',' + dataResampled[i][1] + '\n')
    f.close()
    return

def main():
    # Cache the ui to filename mappings
    uiToFilename = dict()
    cacheUItoFilename(uiToFilename, "./trainingSymbols/")
    cacheUItoFilename(uiToFilename, "./trainingJunk/")

    # Training/Test split by each symbol class
    generateTrainTestSplit("./trainingSymbols/iso_GT.txt", "./trainingSymbols/iso_GT_train.txt", "./trainingSymbols/iso_GT_test.txt", uiToFilename, "./trainingSymbols/iso_GT_test_eval.txt")
    generateTrainTestSplit("./trainingJunk/junk_GT_v3.txt", "./trainingJunk/junk_GT_train.txt",
                           "./trainingJunk/junk_GT_test.txt", uiToFilename, "./trainingJunk/junk_GT_test_eval.txt")

    # Resample the real training symbols for train/test splits
    #generateBalancedClasses("./trainingSymbols/iso_GT_train.txt", "./trainingSymbols/iso_GT_train_resampled.txt")
    #generateBalancedClasses("./trainingSymbols/iso_GT_test.txt", "./trainingSymbols/iso_GT_test_resampled.txt")

    # Resample the junk symbols (no resampling needed)
    # generateBalancedClasses("./trainingJunk/junk_GT.txt", "./trainingJunk/junk_GT_resampled.txt")
    return

main()