/* file: dt_cls_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
!  Content:
!    C++ example of Decision tree classification in the batch processing mode.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DT_CLS_DENSE_BATCH"></a>
 * \example dt_cls_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"
#include <cstdio>

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

/* Input data set parameters */
const std::string trainDatasetFileName = "../data/batch/decision_tree_train.csv";
const std::string pruneDatasetFileName = "../data/batch/decision_tree_prune.csv";
const std::string testDatasetFileName = "../data/batch/decision_tree_test.csv";

const size_t nFeatures = 5; /* Number of features in training and testing data sets */
const size_t nClasses = 5; /* Number of classes */

decision_tree::classification::training::ResultPtr trainingResult;
classifier::prediction::ResultPtr predictionResult;
NumericTablePtr testGroundTruth;

void trainModel();
void testModel();
void printResults();

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 2, &trainDatasetFileName, &testDatasetFileName);

    trainModel();
    testModel();
    printResults();

    return 0;
}

void trainModel() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and labels */
    NumericTablePtr trainData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::notAllocate));
    NumericTablePtr trainGroundTruth(new HomogenNumericTable<>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainGroundTruth));

    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock(mergedData.get());

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the pruning input data from a .csv file */
    FileDataSource<CSVFeatureManager> pruneDataSource(pruneDatasetFileName,
                                                      DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for pruning data and labels */
    NumericTablePtr pruneData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::notAllocate));
    NumericTablePtr pruneGroundTruth(new HomogenNumericTable<>(1, 0, NumericTable::notAllocate));
    NumericTablePtr pruneMergedData(new MergedNumericTable(pruneData, pruneGroundTruth));

    /* Retrieve the data from the pruning input file */
    pruneDataSource.loadDataBlock(pruneMergedData.get());

    /* Create an algorithm object to train the Decision tree model */
    decision_tree::classification::training::Batch<> algorithm(nClasses);

    /* Pass the training data set, labels, and pruning dataset with labels to the algorithm */
    algorithm.input.set(classifier::training::data, trainData);
    algorithm.input.set(classifier::training::labels, trainGroundTruth);
    algorithm.input.set(decision_tree::classification::training::dataForPruning, pruneData);
    algorithm.input.set(decision_tree::classification::training::labelsForPruning,
                        pruneGroundTruth);

    /* Train the Decision tree model */
    algorithm.compute();

    /* Retrieve the results of the training algorithm  */
    trainingResult = algorithm.getResult();
}

void testModel() {
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> testDataSource(testDatasetFileName,
                                                     DataSource::notAllocateNumericTable,
                                                     DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for testing data and labels */
    NumericTablePtr testData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::notAllocate));
    testGroundTruth = NumericTablePtr(new HomogenNumericTable<>(1, 0, NumericTable::notAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(testData, testGroundTruth));

    /* Retrieve the data from input file */
    testDataSource.loadDataBlock(mergedData.get());

    /* Create algorithm objects for Decision tree prediction with the default method */
    decision_tree::classification::prediction::Batch<> algorithm;

    /* Pass the testing data set and trained model to the algorithm */
    algorithm.input.set(classifier::prediction::data, testData);
    algorithm.input.set(classifier::prediction::model,
                        trainingResult->get(classifier::training::model));

    /* Compute prediction results */
    algorithm.compute();

    /* Retrieve algorithm results */
    predictionResult = algorithm.getResult();
}

void printResults() {
    printNumericTables<int, int>(testGroundTruth,
                                 predictionResult->get(classifier::prediction::prediction),
                                 "Ground truth",
                                 "Classification results",
                                 "Decision tree classification results (first 20 observations):",
                                 20);
}
