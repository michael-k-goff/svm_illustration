# Very basic SVM Classifier to illustrate the concept with synthetic data

import random
import pandas as pd
from sklearn import svm

# Build a simple set set to test an SVM classifier. Inputs are x and y values, both dimensions ranging from 0 to 1.
# Outputs are 0 if x^2+y^1 <= 1, and 1 if x^2+y^1 > 1.
def makeSVMTest():
    num_train = 1000
    num_test = 200
    
    x_train = [random.random()*3-1.5 for i in range(num_train)]
    y_train = [random.random()*3-1.5 for i in range(num_train)]
    z_train = [int(x_train[i]*x_train[i]+y_train[i]*y_train[i]>1) for i in range(num_train)]
    df_train = pd.DataFrame(data={"x":x_train,"y":y_train,"z":z_train})
    df_train.to_csv("train_data.csv",index=False)
    
    x_test = [random.random()*3-1.5 for i in range(num_test)]
    y_test = [random.random()*3-1.5 for i in range(num_test)]
    z_test = [int(x_test[i]*x_test[i]+y_test[i]*y_test[i]>1) for i in range(num_test)]
    df_test = pd.DataFrame(data={"x":x_test,"y":y_test,"z":z_test})
    df_test.to_csv("test_data.csv",index=False)

# Build the SVM Classifier on the simple test set
def SVMClassifier():
    df_train = pd.read_csv("train_data.csv")
    X_train = df_train[["x","y"]]
    y_train = df_train["z"]
    clf = svm.SVC()
    clf = clf.fit(X_train,y_train)
    return clf
    
# Evaluate the SVM Classifier from the simple test set
# Return a dataframe of errors.
# If there are any errors, they should be points for which x^2+y^2 is close to 1. In other words, close to the boundary between the classes.
def SVMEval(clf):
    df_test = pd.read_csv("test_data.csv")
    X_test = df_test[["x","y"]]
    y_pred = clf.predict(X_test)
    df_test["pred"] = y_pred
    errors = df_test[df_test["z"] != df_test["pred"]]
    print "Accuracy: " + str(1-float(len(errors))/len(df_test))
    return errors
    
# Build out the full data set, build the classifier, and evaluate it on the test set
def BuildAndEvaluate():
    makeSVMTest()
    clf = SVMClassifier()
    errors = SVMEval(clf)
    return errors