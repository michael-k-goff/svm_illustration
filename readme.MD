# Support Vector Machine illustration

This project illustrates, with three examples, simple use cases for support vector machines (SVM).

To run this project, you will need to download and install the [MNIST handwritten digit data set](http://yann.lecun.com/exdb/mnist/) and place the files in the same directory as the code. You will also need the latest version (I am using 0.19.1) of sklearn.

### Classifying Synthetic Data

The first use case shows how an out-of-the-box SVM model separates regions in the plane. The data is a binary classification (0 or 1) with 0's inside the circle x^2+y^2 = 1, with points inside the circle classified as 0 and points outside the circle classified as 1. No linear classifier could perform well in learning the distiction. By contrast, the SVM model gets at least 96% accuracy in general on a modest (1000 examples) training set.

### MNIST Digit Set

We next apply the SVM model to something a bit less trivial. In our earlier [neural network](https://github.com/pepper2000/neural_net), we found that our custom neural net, without significant optimization or hyperparameter tuning, scored about 90% in classifying handwritten digits. Here we apply the same transformations to the data set as in the neural network project.

Training nonlinear SVM models can be [quite slow](https://www.quora.com/What-is-the-computational-complexity-of-an-SVM). This project only uses 10,000 of the 60,000 training examples in the MNIST data set. Presumably the performance would be better if we used the full training set. As implemented, using 10,000 training examples and no parameter tuning, the SVM classifier has an accuracy of about 92%.

### SVM Regressor

Our third use case also uses synthetic data, with the response variable given as an algebraic function of 10 independent variables. Model performance is assessed by the mean square effort of the predicted values. We apply two versions of the SVM regressor. One is the out-of-the-box model with default parameters, and the other is tuned through a grid search over several parameter options, as indicated [here](https://github.com/ksopyla/svm_mnist_digit_classification). The MSE of the models is highly variable and dependent on how the random number generator gives the underlying function, but in general the grid search algorithm performs significantly better than the untuned version.