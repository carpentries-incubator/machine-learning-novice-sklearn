#!/usr/bin/env python3

import sklearn.datasets as skl_data
import sklearn.neural_network as skl_nn
import sklearn.model_selection as skl_msel


print("loading data")

# Load data from https://www.openml.org/d/554
data, labels = skl_data.fetch_openml('mnist_784', version=1, return_X_y=True)
data = data / 255.0


mlp = skl_nn.MLPClassifier(hidden_layer_sizes=(50,), max_iter=50, random_state=1)

kfold = skl_msel.KFold(4)
# enumerate loops through a list and gives us an index and a value
# in this case it gives us a pair of values
for (train, test) in kfold.split(data):
    print("Training data",train[0],train[-1])
    X_train = data[train]
    y_train = labels[train]
    print("Test data",test[0],test[-1])
    X_test = data[test]
    y_test = labels[test]
    mlp.fit(X_train,y_train)
    print("Training set score", mlp.score(X_train, y_train))
    print("Testing set score", mlp.score(X_test, y_test))
