# Perform an SVM regression on some synthetic data
# This illustrates SVM regression and grid search over a parameter space

import pandas as pd
import numpy as np
import random
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# Generate synthetic data. The inputs x_i are chosen randomly from [0,1]
# The output is Sum coef_i x_i^(exp_i), where coef_i is a randomly generated coefficient in [-1,1]
# and exp_i is a randomly generated exponent in [0,2].
# By default, there are num_x=10 input features
num_x = 10
def makeSVMTest2():
    num_train = 1000
    num_test = 200
    X_train = [[random.random() for j in range(num_x)] for i in range(num_train)]
    coef = coef = [random.random()*2-1 for i in range(num_x)]
    exp = [random.random()*2 for i in range(num_x)]
    y_train = [sum([coef[j]*X_train[i][j]**exp[j] for j in range(num_x)]) for i in range(num_train)]
    data_dict = {}
    for i in range(num_x):
        data_dict["x"+str(i)] = [X_train[j][i] for j in range(num_train)]
    data_dict["y"] = y_train
    df_train = pd.DataFrame(data=data_dict)
    df_train.to_csv("train_data2.csv",index=False)
    
    X_test = [[random.random() for j in range(num_x)] for i in range(num_test)]
    y_test = [sum([coef[j]*X_test[i][j]**exp[j] for j in range(num_x)]) for i in range(num_test)]
    data_dict = {}
    for i in range(num_x):
        data_dict["x"+str(i)] = [X_test[j][i] for j in range(num_test)]
    data_dict["y"] = y_test
    df_test = pd.DataFrame(data=data_dict)
    df_test.to_csv("test_data2.csv",index=False)
    
# Build the SVM Classifier on the above test set
def SVMRegressor():
    df_train = pd.read_csv("train_data2.csv")
    X_train = df_train[["x"+str(i) for i in range(num_x)]]
    y_train = df_train["y"]
    clf = svm.SVR()
    # Grid Search. See https://github.com/ksopyla/svm_mnist_digit_classification for source of this code
    gamma_range = np.outer(np.logspace(-3, 0, 4),np.array([1,5]))
    gamma_range = gamma_range.flatten()
    C_range = np.outer(np.logspace(-1, 1, 3),np.array([1,5]))
    C_range = C_range.flatten()
    parameters = {'kernel':['rbf'], 'C':C_range, 'gamma': gamma_range}
    grid_clsf = GridSearchCV(estimator=clf,param_grid=parameters,n_jobs=1, verbose=2)
    grid_clsf.fit(X_train, y_train)
    # For comparison, this is a SVM classifier using fully default parameters
    clf = clf.fit(X_train,y_train)
    return clf, grid_clsf
    
# Evaluate the SVM Regressor from the above test set
def SVMRegressorEval(clf):
    df_test = pd.read_csv("test_data2.csv")
    X_test = df_test[["x"+str(i) for i in range(num_x)]]
    y_test = df_test["y"]
    y_pred = clf.predict(X_test)
    y_var = np.var(y_test)
    mse = sum([(y_test[i]-y_pred[i])**2])/len(y_test)
    # Sanity checks
    df_test["predicted"] = y_pred
    return mse/y_var
    
def BuildAndEvaluateSVR():
    makeSVMTest2()
    clf, grid_clsf = SVMRegressor()
    score_clf = SVMRegressorEval(clf)
    score_grid = SVMRegressorEval(grid_clsf)
    return {"Default SVM Score":score_clf, "Grid Search Score":score_grid}