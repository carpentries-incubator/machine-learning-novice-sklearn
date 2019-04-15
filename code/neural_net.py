#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 00:31:03 2019

@author: colin
"""

import sklearn.datasets as skl_data
import sklearn.neural_network as skl_nn

print("loading data")
data, labels = skl_data.fetch_openml('mnist_784', version=1, return_X_y=True)
data = data / 255.0


mlp = skl_nn.MLPClassifier(hidden_layer_sizes=(80,), max_iter=50, 
                 verbose = True, random_state=1)

data_train = data[0:50000]
labels_train = labels[0:50000]

data_test = data[50001:]
labels_test = labels[50001:]

mlp.fit(data_train,labels_train)
print("Training set score", mlp.score(data_train, labels_train))
print("Testing set score", mlp.score(data_test, labels_test))
    
# opencv not installed by default
# conda install -c conda-forge opencv  
#import cv2
#import matplotlib.pyplot as plt
#test_digit = cv2.imread("digit.png")[:,:,0]
#test_digit = 1.0 - (test_digit/255.0)
#plt.matshow(test_digit)
#plt.show()
#print(mlp.predict([test_digit.reshape(784)]))
