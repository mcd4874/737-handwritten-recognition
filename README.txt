README.txt

CSCI-737 Project 2
Authors: William Duong and Eric Hartman

Project 2 Files & Updates:
--------------------------
datasplit.py                    - Splits the dataset into a 70% train and 30% test split by using
                                  a greedy algorithm that builds up these splits by minimizing variance.

                                  Usage:
                                    datasplit.py <inputFile> <trainFile> <testFile>

                                    <inputFile> contains a list of .inkml filenames with full paths
                                    <trainFile> will contain a list of files for the train split
                                    <testFile> will contain a list of files for the test split

segmenter.py                    - Performs segmentation on the inputFile by applying either our
                                  sophisticated segmenter ("real") or the baseline segmenter ("baseline") to
                                  produce a label graph output file.  Places these output .lg files in the
                                  same path as the source .inkml files

                                  Usage:
                                    segmenter.py <real|baseline> <inputFile>

                                    <real|baseline>  Optional: Specify 'real' to use sophisticated segmenter on a list of files, OR...
                                                               Specify 'baseline' to use baseline segmenter on a list of files
                                    <inputFile>      Either an .inkml file to be segmented or a file containing a list of .inkml files

generateFeatureStack.py (       - Updated since Project 1 to incorporate additional features used by classifier.
generateNormalizedImages_v2.py  - Updated since Project 1 to incorporate additional features used by classifier.
testClassifiers_v2.py           - Updated since Project 1 to incorporate additional features used by classifier.
trainClassifiers.py             - Updated since Project 1 to incorporate additional features used by classifier.


Project 1 Files:
----------------
generateBalancedClasses.py      - Constructs train/test splits of ground truth files
                                - Training split is in format consumed by generateNormalizedImages_v2.py
                                - Test split is in format consumed by testClassifiers_v2.py

generateNormalizedImages_v2.py  - Parses all inkml files in ./trainingSymbols and ./trainingJunk
                                  converting the files into stacks of feature vectors as follows:
                                    Creates "./symbolStack.csv" = feature vectors for all trainingSymbols
                                    Creates "./junkStack.csv"   = feature vectors for all trainingJunk

                                  These "stack" files are used by the trainClassifiers.py program to
                                  train the 4 classifier moddels.

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
                                    processed into features, as well as mapping functionality to translate
                                    between UI and filenames.

trainClassifiers.py             - Trains the kdtree and randomforest classifiers for the the valid & valid+junk datasets
                                  Ultimately, 4 trained classifier models are output.

                                  Contains hyperparameters for configuration of randomforest:
                                    maxTrees = maximum number of trees in the random forest
                                    maxDepth = maximum depth of trees in the random forest

                                  Creates pickled models for use by testClassifiers_v2.py
                                    encoder_rf.pkl = classes encoder for random forest symbols model
                                    pickle_rf.pkl = trained random forest symbols model
                                    encoder_kD.pkl = classes encoder for kdtree symbols model
                                    pickle_kd.pkl = trained kdtree symbols model
                                    encoder_both_rf.pkl = classes encoder for random forest combined model
                                    pickle_both_rf.pkl = trained random forest combined model
                                    encoder_both_kD.pkl = classes encoder for kdtree combined model
                                    pickle_both_kd.pkl = trained kdtree combined model

testClassifiers_v2.py           -  Tests the pickled kdtree and randomforest classifiers according to the commandline
                                    inputs that are provided.  Input file syntax is according to assignment instructions.

                                    Usage:
                                        testClassifiers.py <inputFile> <outputFile> <classifierIdentifier>

                                        <inputFile> contains a list of .inkml filenames
                                        <outputFile> output csv file containing UI, Top10 recognitions
                                        <classifierIdentifier> is kdtree or randomforest

                                    Example:
                                        testClassifiers.py ./trainingSymbols/iso_GT_test.txt randomForestSymbols-output.csv randomForestSymbols
                                        testClassifiers.py ./trainingSymbols/combined_GT_test.txt randomForestCombined-output.csv randomForestCombined
                                        testClassifiers.py ./trainingSymbols/iso_GT_test.txt kdtreeSymbols-output.csv kdtreeSymbols
                                        testClassifiers.py ./trainingSymbols/combined_GT_test.txt kdtreeCombined-output.csv kdtreeCombined

                                    Creates csv outputFile in the form of "<UI>, <Top10-descending-order>"


How To...
---------
How to run a classifier for a given ground truth file?
    You only need to run the testClassifiers_v2.py as documented above.

    For example, to run a test using a ground truth file named "iso_GT_test.txt" against randomFroestSymbols model:
    # testClassifiers.py ./trainingSymbols/iso_GT_test.txt randomForestSymbols-output.csv randomForestSymbols

How to rebuild the classifier models?
    Prerequisite: "task2-trainSymb2014(1).zip" is unpacked to the working directory such that ./trainingJunk
    and ./trainingSymbols are accessible.

    To rebuild the classifier models, it is simply a matter of re-running a sequence of programs in the following order.
    Note: There are no command-line parameters needed.  These programs were validated on Python 3 environment only.

    # generateBalancedClasses.py
    # generateNormalizedImages_v2.py
    # trainClassifiers.py

    After running these three programs, all of the required models will have been built and the environment
    will be ready for testClassifiers_v2.py to run test datasets.

How to perform train/test split?
    Prerequisite: A file is created that contains the listing of all .inkml files in the complete dataset.
    This can be constructed through usage of ls & cat unix commands.

    To perform the 70/30 train/test split, run the following:
    # datasplit.py allfiles.out trainfiles.out testfiles.out

    After running, the "trainfiles.out" will contain the list of inkml files for the train dataset and
    the "testfiles.out" will contain the list of inkml files for the test dataset.

How to segment a particular inkml file?
    You only need to run the segment.py as documented above.

    For example, to segment a file named "./Train/inkml/all/2009213-137-177.inkml":
    # segment.py real ./Train/inkml/all/2009213-137-177.inkml

    This will produce an output file: ./Train/inkml/all/2009213-137-177.lg

How to segment a list of inkml file?
    You only need to run the segment.py as documented above.

    For example, to segment a list of .inkml files contained in a file named "testfiles.out":
    # segment.py real testfiles.out


