# This is a simple illustration of support vector machines (SVM). We conduct three analyses, two with classifiers and one with a regressor.

import os

os.chdir('My Path Here')

import simple_svm
reload(simple_svm)
import mnist_svm
reload(mnist_svm)
import svr
reload(svr)

# Part 1: Simple SVM
# This function will print out the accuracy of the simple SVM model on the test set
# On the generate synthetic data set, accuracy is usually above 96%.
simple_svm_errors = simple_svm.BuildAndEvaluate()

# Build and test a basic SVM model on the MNIST digit set, return the score, the proportion of digits classified correctly
mnist_score = mnist_svm.BuildAndEvaluateMNIST()

# Build and test an SVM regressor on synthetic data.
# There tends to be wide variance in the scores, as the synthetic data is regenerated every time, but in general, the grid search performs significantly better
# The score is given as the ratio of the mean square error of predictions to variance of the test values.
# Thus a score of 0 would indicate perfect prediction, while a score of 1 would be equivalent to guessing the test set's mean for all cases.
svr_score = svr.BuildAndEvaluateSVR()