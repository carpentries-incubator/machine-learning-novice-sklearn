---
title: "Neural Networks"
teaching: 0
exercises: 0
questions:
- "Key question (FIXME)"
objectives:
- "Train a neural network"
- "Understand cross validation"
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---
FIXME

# Introduction

Neural networks are a machine learning method inspired by how the human brain works. They are particularly good at doing pattern recognition and classification tasks, often using images as inputs. They're a well established machine learning technique that has been around since the 1950s but have gone through several iterations since that have overcome fundamental limitations of the previous one. The current state of the art neural networks are often referred to as deep learning 


## Perceptrons

Perceptrons are the building blocks of neural networks, they're an artificial version of a single neuron on the brain. They typically have one or more inputs and a single output. Each input will be multiplied by a weight and the value of all the weighted inputs are then summed together. Finally the summed value is put through an activation function which decides if the neuron "fires" a signal. In some cases this activation function is simply a threshold step function which outputs zero below a certain input and one above it. Other designs of neurons use other activation functions, but typically they have an output between zero and one and are still step like in their nature.

### Coding a perceptron 

Inputs is a list of input values. 
Weights is a list of weight values.
Threshold is that activation threshold.

First lets multiply each input by the corresponding weight. Do to this quickly and conciesely we'll use the numpy multiply function which can multiply each item in a list by a corresponding item in another list. 

We then take the sum of all the inputs multiplied by their weights. Finally if this value is less than the activation threshold we output z ero otherwise we output a one.

~~~
import numpy as np
def perceptron(inputs, weights, threshold):
    
    assert len(inputs) == len(weights)

    # multiply the inputs and weights
    values = np.multiply(inputs,weights)
    
    # sum the results 
    total = sum(values)

    # decide if we should activate the perceptron
    if total < threshold:
        return 0
    else:
        return 1
~~~
{: .python}

### Computing with a perceptron

A single perceptron can perform basic linear classifcation problems such as computing the logical AND, OR and NOT functions.

OR 

| Input 1 | Input 2 | Output |
| --------|---------|--------|
| 0       |0        |0       |
| 0       |1        |1       |
| 1       |0        |1       |
| 1       |1        |1       |

AND

| Input 1 | Input 2 | Output |
| --------|---------|--------|
| 0       |0        |0       |
| 0       |1        |0       |
| 1       |0        |0       |
| 1       |1        |1       |


NOT

| Input 1 |Output |
| --------|--------|
| 0       |1       |
| 1       |0       |


We can get a single perceptron to compute each of these functions

OR:
~~~
inputs = [[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]]
for input in inputs:
    print(input,perceptron(input, [0.5,0.5], 0.5))
~~~
{: .python}


AND:
~~~
inputs = [[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]]
for input in inputs:
    print(input,perceptron(input, [0.5,0.5], 1.0))
~~~
{: .python}


NOT:

The NOT function only has a single input but to make it work in the perceptron we need to introduce a bias term which is always the same value. In this example its the second input. This has a weight of 1.0, the weight on the real input is -1.0. 
~~~
inputs = [[0.0,1.0],[1.0,1.0]]
for input in inputs:
    print(input,perceptron(input, [-1.0,1.0], 1.0))
~~~
{: .python}


A perceptron can be trained to compute any function which is has linear seperability. A simple training algorithm called the perceptron learning algorithm can be used to do this and scikit-learn has its own implementation of it. 

But alone a single perceptron cannot be used to solve a non-linearly seperable function. For that we need to use multiple perceptrons. 

## Network of perceptrons

They are formed of networks of artificial neurons which each take one or more inputs and typically have a single output. The neurons are connected together in large networks typically of 10s to 1000s of neurons. Typically networks are connected in layers with an input layer, middle or hidden layer (or layers) and finally an output layer. 


## Cross Validation

Previously we split the data into training and test sets. But what happens if the test set includes important features we want to train on that happen to be missing in the test set? 
By doing this we're reducing the amount of data we can use for training. 

Cross validation runs the training/testing multiple times but splits the data in a different way each time. This way all of the data gets used both for training and testing. We can then use the entire dataset to train after (assuming cross validation succeeds)

example list

[1,2,3,4,5,6,7,8]

train = 1,2,3,4,5,6
test = 7,8

train = 1,2,3,4,7,8
test = 5,6

train = 1,2,5,6,7,8
test = 3,4

train = 3,4,5,6,7,8
test = 1,2

(generate an image of this)

### Cross Validation code example


## Deep Learning

Deep learning usually refers to newer neural network architectures which use a special type of network known as a convolutional network. Typically these have many layers and thousands of neurons. They're very good at tasks such as image recognition but take a long time to train and run. They're often used with GPU (Graphical Processing Units) which are good at executing multiple operations simultaneously. 
Its very common to use cloud computing or HPC systems with multiple GPUs attached. 

Scikit learn isn't really setup for Deep Learning and we'll have to rely on other libraries. Common choices include Google's TensorFlow, Keras or Torch. There is however an interface layer between sklearn and tensorflow called skflow. Below is a short example using skflow.

https://www.kdnuggets.com/2016/02/scikit-flow-easy-deep-learning-tensorflow-scikit-learn.html
