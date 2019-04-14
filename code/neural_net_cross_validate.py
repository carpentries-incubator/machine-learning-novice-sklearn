#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 22:02:19 2019

@author: colin
"""

import matplotlib.pyplot as plt
import sklearn.datasets as skl_data
import sklearn.neural_network as skl_nn
import sklearn.model_selection as skl_msel


print("loading data")

# Load data from https://www.openml.org/d/554
X, y = skl_data.fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.0


mlp = skl_nn.MLPClassifier(hidden_layer_sizes=(50,), max_iter=50, alpha=1e-4,
                solver='sgd', tol=1e-4, random_state=1,
                learning_rate_init=.1)
# verbose=10, 

kfold = skl_msel.KFold(4)
# enumerate loops through a list and gives us an index and a value
# in this case it gives us a pair of values
for counter, (train, test) in enumerate(kfold.split(X)):
    X_train = X[train]
    y_train = y[train]
    
    X_test = X[test]
    y_test = y[test]
    mlp.fit(X_train,y_train)
    print("Training set score", mlp.score(X_train, y_train))
    print("Testing set score", mlp.score(X_test, y_test))
    
# opencv not installed by default
# conda install -c conda-forge opencv  
#import cv2
#test_digit = cv2.imread("digit.png")[:,:,0]
#test_digit = 1.0 - (test_digit/255.0)
#plt.matshow(test_digit)
#plt.show()
#print(mlp.predict([test_digit.reshape(784)]))
