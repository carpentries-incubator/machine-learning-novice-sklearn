#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:45:45 2019

@author: colin
"""
import numpy as np

def perceptron(inputs, weights, threshold):

    values = np.multiply(inputs,weights)

    total = sum(values)

    if total < threshold:
        return 0
    else:
        return 1


#OR
print("Computing OR")
inputs = [[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]]
for input in inputs:
    print(input,perceptron(input, [0.5,0.5], 0.5))

#NOR
print("Computing NOR")
inputs = [[0.0,0.0,1.0],[1.0,0.0,1.0],[0.0,1.0,1.0],[1.0,1.0,1.0]]
for input in inputs:
    print(input,perceptron(input, [-0.5,-0.5,1.0], 0.5))


#AND
inputs = [[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]]
for input in inputs:
    print(input,perceptron(input, [0.5,0.5], 1.0))

#NAND:
print("Computing NAND")
inputs = [[0.0,0.0,1.0],[1.0,0.0,1.0],[0.0,1.0,1.0],[1.0,1.0,1.0]]
for input in inputs:
    print(input,perceptron(input, [-0.5,-0.5,1.0], 1.0))

#NOT
print("Computing NOT")
inputs = [[0.0,1.0],[1.0,1.0]]
for input in inputs:
    print(input,perceptron(input, [-1.0,1.0], 1.0))