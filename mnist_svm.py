# Train an SVM Classifier on the MNIST digit set

import idx2numpy
from sklearn import svm

# Train and test an SVM on the MNIST digit set
def SVMClassifierMNIST():
    # We restrict to 10000 data points because the full 60,000 take inordinately long to train the model.
    # To use the full training set, set to 60000 and wait for a while. Try a smaller value for faster execution
    num_train = 10000
    train_X = idx2numpy.convert_from_file("../Project 2 - Neural Network/train-images-idx3-ubyte")
    # It is necessary to rescale the input data to be between 0 and 1 for the model to work
    train_X = train_X.reshape(train_X.shape[0],train_X.shape[1]*train_X.shape[2]) / 256.
    train_y = idx2numpy.convert_from_file("../Project 2 - Neural Network/train-labels-idx1-ubyte")
    train_XX = train_X[0:num_train]
    train_yy = train_y[0:num_train]
    clf = svm.SVC()
    clf = clf.fit(train_XX,train_yy)
    return clf
    
def MNIST_SVMEval(clf):
    test_X = idx2numpy.convert_from_file("t10k-images-idx3-ubyte")
    # Note: this is the same transformation applied to the training set. It must be the same.
    test_X = test_X.reshape(test_X.shape[0],test_X.shape[1]*test_X.shape[2]) / 256.
    pred_y = clf.predict(test_X)
    return pred_y
    
def MNIST_SVM_Score(pred_y):
    test_y = idx2numpy.convert_from_file("t10k-labels-idx1-ubyte")
    # The score is the proportion of the test digits identified correctly
    return sum(pred_y == test_y) / float(len(test_y))
    
def BuildAndEvaluateMNIST():
    clf = SVMClassifierMNIST()
    pred_y = MNIST_SVMEval(clf)
    score = MNIST_SVM_Score(pred_y)
    print "The SVM model got " + str(score) + " of the digits right."
    return score