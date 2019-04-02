README.txt

CSCI-737 Project 1
Authors: William Duong and Eric Hartman

Files:
generateBalancedClasses.py      - Constructs train/test splits of ground truth files
                                - Training split is in format consumed by generateNormalizedImages_v2.py
                                - Test split is in format consumed by testClassifiers_v2.py

generateNormalizedImages_v2.py  - Parses all inkml files in ./trainingSymbols and ./trainingJunk
                                  converting the files into stacks of feature vectors as follows:
                                    Creates "./symbolStack.csv" = feature vectors for all trainingSymbols
                                    Creates "./junkStack.csv"   = feature vectors for all trainingJunk

                                  This file contains some of the hyperparameters for tuning the features,
                                  such as:
                                    w = standardized width of image
                                    h = standardized height of image
                                    thickness = line thickness when generating images of samples
                                    sector = angular resolution for normalizing angles between consecutive points
                                    binCount = histogram bin count for angles between consecutive points
                                    featureCount = total number of angles between consecutive points

generateFeatureStack.py        - Library file containing helper methods used by testClassifiers_v2.py and
                                    generateBalancedClasses.py.

                                  This file contains helper functions that allow for test input files to be
                                    processed into feature , as well as mapping functionality to translate
                                    between UI and filenames.

trainClassifiers.py             - Trains the kdtree and randomforest classifiers for the the valid & valid+junk datasets

                                  Contains hyperparameters for configuration of randomforest:
                                    maxTrees = maximum number of trees in the random forest
                                    maxDepth = maximum depth of trees in the random forest

                                  Creates pickled models for use by testClassifiers_v2.py
                                    pickle_encoder.pkl = classes encoder
                                    pickle_kdtree.pkl = trained kdtree model
                                    pickle_rf.pkl = trained random forest model

testClassifiers_v2.py           -  Tests the pickled kdtree and randomforest classifiers according to the commandline
                                    inputs that are provided.  Input file syntax is according to assignment instructions.

                                    Usage:
                                        testClassifiers.py <inputFile> <outputFile> <classifierIdentifier>

                                        <inputFile> contains a list of .inkml filenames
                                        <outputFile> output csv file containing UI, Top10 recognitions
                                        <classifierIdentifier> is kdtree or randomforest

                                    Example:
                                        testClassifiers.py ./trainingSymbols/iso_GT_test.txt randomForestSymbols-output.csv randomForestSymbols
                                        testClassifiers.py ./trainingSymbols/iso_GT_combined.txt randomForestCombined-output.csv randomForestCombined
                                        testClassifiers.py ./trainingSymbols/iso_GT_test.txt kdtreeSymbols-output.csv kdtreeSymbols
                                        testClassifiers.py ./trainingSymbols/iso_GT_combined.txt kdtreeCombined-output.csv kdtreeCombined


                                    Creates outputFile in the form of "<UI>, <Top10-descending-order>"






