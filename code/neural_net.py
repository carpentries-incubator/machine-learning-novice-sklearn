#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 00:31:03 2019

@author: colin
"""

import matplotlib.pyplot as plt
import sklearn.datasets as skl_data
import sklearn.neural_network as skl_nn


print("loading data")

# Load data from https://www.openml.org/d/554
X, y = skl_data.fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.0


mlp = skl_nn.MLPClassifier(hidden_layer_sizes=(50,), max_iter=50, alpha=1e-4,
                solver='sgd', tol=1e-4, random_state=1,
                learning_rate_init=.1)

X_train = X[0:50000]
y_train = y[0:50000]

X_test = X[50001:]
y_test = y[50001:]

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
