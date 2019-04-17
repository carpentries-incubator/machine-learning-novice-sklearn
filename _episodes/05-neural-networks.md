---
title: "Neural Networks"
teaching: 20
exercises: 30
questions:
- "How can we classify images using a neural network?"
objectives:
- "Understand how a single artificial neuron (perceptron) works"
- "Understand that layers of perceptrons allow non-linear separable problems to be solved"
- "Train a neural network"
- "Understand cross validation"
keypoints:
- "Perceptrons are artificial neurons which build neural networks."
- "A perceptron takes multiple inputs, multiplies each by a weight value and sums the weighted inputs. It then applies an activation function to the sum."
- "A single perceptron can solve simple functions which are linearly separable."
- "Multiple perceptrons can be combined to form a neural network which can solve functions that aren't linearly separable."
- "We can train a whole neural network with the back propagation algorithm. Scikit-learn includes an implementation of this algorithm."
- "Training a neural network requires some training data to show the network examples of what to learn."
- "To validate our training we split the the training data into a training set and a test set."
- "To ensure the whole dataset can be used in training and testing we can train multiple times with different subsets of the data acting as training/testing data. This is called cross validation."
- "Deep learning neural networks are a very powerful modern technique. Scikit learn doesn't support these but other libraries like Tensorflow do."
- "Several companies now offer cloud APIs where we can train neural networks on powerful computers."
---


# Introduction

Neural networks are a machine learning method inspired by how the human brain works. They are particularly good at doing pattern recognition and classification tasks, often using images as inputs. They're a well established machine learning technique that has been around since the 1950s but have gone through several iterations since that have overcome fundamental limitations of the previous one. The current state of the art neural networks are often referred to as deep learning 


## Perceptrons

Perceptrons are the building blocks of neural networks, they're an artificial version of a single neuron on the brain. They typically have one or more inputs and a single output. Each input will be multiplied by a weight and the value of all the weighted inputs are then summed together. Finally the summed value is put through an activation function which decides if the neuron "fires" a signal. In some cases this activation function is simply a threshold step function which outputs zero below a certain input and one above it. Other designs of neurons use other activation functions, but typically they have an output between zero and one and are still step like in their nature.

![A diagram of a perceptron](../fig/perceptron.svg)

### Coding a perceptron 

Below is an example of a perceptron written as a Python function. The function takes three parameters: Inputs is a list of input values, Weights is a list of weight values and Threshold is the activation threshold.

First lets multiply each input by the corresponding weight. Do to this quickly and concisely we'll use the numpy multiply function which can multiply each item in a list by a corresponding item in another list. 

We then take the sum of all the inputs multiplied by their weights. Finally if this value is less than the activation threshold we output zero otherwise we output a one.

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

A single perceptron can perform basic linear classification problems such as computing the logical AND, OR and NOT functions.

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

A perceptron can be trained to compute any function which is has linear separability. A simple training algorithm called the perceptron learning algorithm can be used to do this and scikit-learn has its own implementation of it. We're going to skip over the perceptron learning algorithm and move straight onto more powerful techniques. If you want to learn more about it see [this page](https://computing.dcu.ie/~humphrys/Notes/Neural/single.neural.html) from Dublin City University. 

> # Building a perceptron for NAND
> Try and modify the perceptron examples above to calculate the NAND function. 
> This is the inverse of the AND function and its truth table looks like:
>
> | Input 1 | Input 2 | Output |
> | --------|---------|--------|
> | 0       |0        |1       |
> | 0       |1        |1       |
> | 1       |0        |1       |
> | 1       |1        |0       |
>
>> #Solution
>>~~~
>>inputs = [[0.0,1.0],[1.0,1.0]]
>>for input in inputs:
>>   print(input,perceptron(input, [-1.0,1.0], 1.0))
>>~~~
{: .python}

### Perceptron Limitations

A single perceptron cannot solve any function that is not linearly separable, meaning that we need to be able to divide the classes of inputs and outputs with a straight line. A common example of this is the XOR function shown below:

| Input 1 | Input 2 | Output |
| --------|---------|--------|
| 0       |0        |0       |
| 0       |1        |1       |
| 1       |0        |1       |
| 1       |1        |0       |

(Make a graph of this)

This function outputs a zero both when all its inputs are one or zero and its not possible to separate with a straight line. This is known as linear separability, when this limitation was discovered in the 1960s it effectively halted development of neural networks for over a decade in a period known as the "AI Winter".


## Multi Layer Perceptrons

A single perceptron cannot be used to solve a non-linearly separable function. For that we need to use multiple perceptrons and typically multiple layers of perceptrons. They are formed of networks of artificial neurons which each take one or more inputs and typically have a single output. The neurons are connected together in large networks typically of 10s to 1000s of neurons. Typically networks are connected in layers with an input layer, middle or hidden layer (or layers) and finally an output layer. 

![A multilayer perceptron](../fig/multilayer_perceptron.svg)

### Training multi layer perceptrons 

Multilayer perceptrons need to be trained by showing them a set of training data and measuring the error between the network's predicted output and the true value. Training takes an iterative approach that improves the network a little each time a new training example is presented. There are a number of training algorithms available for a neural network today, but we are going to use one of the best established and well known, the backpropagation algorithm. The algorithm is called back propagation because it takes the error calculated between an output of the network and the true value and takes it back through the network to update the weights. If you want to read more about back propagation see [this chapter](http://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf) from the book "Neural Networks - A Systematic Introduction". 

### Multilayer perceptrons in scikit-learn

We're going to build a multilayer perceptron for recognising handwriting from images. Scikit Learn includes some example handwriting data from the [MNIST data set](http://yann.lecun.com/exdb/mnist/), this consists of 70,000 images of hand written digits. Each image is 28x28 pixels in size (784 pixels in total) and is represented in grayscale with values between zero for fully black and 255 for fully white. This means we'll need 784 perceptrons in our input layer, each taking the input of one pixel and 10 in our output layer to represent each digit we might classify. If trained correctly  then only the perceptron in the output layer to "fire" will be on the one representing the in the image (this is a massive oversimplification!). 

We can import this dataset by doing `import sklearn.datasets as skl_data` and then load it into memory with `X, y = skl_data.fetch_openml('mnist_784', version=1, return_X_y=True)`. This creates two arrays of data, one called `data` contains the image data and the other `labels` contains the labels for those data which will tell us which digit is in the image. 

As neural networks typically want to work with data that ranges between 0 and 1.0 we need to normalise our data to this range. Python has a shortcut which lets us divide the entire data array by 255 and store the result, we can simply do `data = data / 255.0` instead of writing a loop ourselves to divide every pixel by 255. Although the final result is the same and will take the same (or very similar) amount of computation. 

Now we need to initalise a neural network, scikit learn has an entire library `sklearn.neural_network` for this and the `MLPClassifier` class handles multilayer perceptrons. This network takes a few parameters including the size of the hidden layer, the maximum number of training iterations we're going to allow, the exact algorithm to use, if we'd like verbose output about what the training is doing and the initial state of the random number generator.

In this example we specify a multilayer perceptron with 50 hidden nodes, we allow a maximum of 50 iterations to train it, we turn on verbose output to see what's happening and initalise the random state to 1 so that we always get the same behaviour. 

`mlp = skl_nn.MLPClassifier(hidden_layer_sizes=(50,), max_iter=50, verbose=1, random_state=1)`

We now have a neural network but we haven't done any training of it yet. Before training lets split our dataset into two halves a training set which we'll use to train the classifier and a test set which we'll use to see how well the training worked. By using different data for the two we can help show we haven't only trained a network which works just with the data it was trained on. We'll take the first 60,000 samples in the dataset as the training data
and the last 10000 as the test data. 

~~~
data_train = data[0:60000]
labels_train = labels[0:60000]
data_test = X[60001:]
labels_test = y[60001:]
~~~
{: .python}

Now lets go ahead and train the network, this line will take about one minute to run. We do this by calling the `fit` function inside the mlp class instance, this needs two arguments the data itself and the labels showing what class each item should be classified to. `mlp.fit(data_train,labels_train)`

Finally lets score the accuracy of our network against both the original training data and the test data. If the training had converged to the point where each iteration of training wasn't improving the accuracy then the accuracy of the training data should be 1.0 (100%). 

~~~
print("Training set score", mlp.score(data_train, labels_train))
print("Testing set score", mlp.score(data_test, labels_test))
~~~
{: .python}

Here is the complete program:


~~~
import matplotlib.pyplot as plt
import sklearn.datasets as skl_data
import sklearn.neural_network as skl_nn

data, labels = skl_data.fetch_openml('mnist_784', version=1, return_X_y=True)
data = data / 255.0


mlp = skl_nn.MLPClassifier(hidden_layer_sizes=(50,), max_iter=50, verbose=1, random_state=1)

data_train = data[0:60000]
labels_train = labels[0:60000]

data_test = data[60001:]
labels_test = labels[60001:]

mlp.fit(data_train, labels_train)
print("Training set score", mlp.score(data_train, labels_train))
print("Testing set score", mlp.score(data_test, labels_test))
~~~
{: .python}


> # Changing the learning parameters
> There are several parameters which control the training of the data. One of these is called the learning rate, increasing this can reduce how many learning iterations we need. But make it too large and we'll end up overshooting.
> Try tweaking this parameter by adding the parameter `learning_rate_init`, the default value of this is 0.001. Try increasing it to around 0.1
{: .challenge}


> # Using your own handwriting
> Create an image using Microsoft Paint, the GNU Image Manipulation Project (GIMP) or [jspaint](https://jspaint.app/). The image needs to be greyscale and 28 x 28 pixels.
> Try and draw a digit (0-9) in the image and save it into your code directory.
> The code below loads the image (called digit.png, change to whatever your file is called) using the OpenCV library. Some Anaconda installations need this installed either through the package manager or by running the command: `conda install -c conda-forge opencv ` from the anaconda terminal.
> OpenCV assumes that images are 3 channel red, green, blue and we have to convert to one channel greyscale with cvtColor.
> We also need to normalise the image by dividing each pixel by 255.
> To verify the image we can plot it by using plt.matshow.
> To check what digit it is we can pass it into mlp.predict, but we have to convert it from a 28x28 array to a one dimensional 784 byte long array with the reshape function.
> Did it correctly classify your hand(mouse) writing? Try a few images. 
> If you have time try drawing images on a touch screen or taking a photo of something you've really written by hand. Remember that you'll have to resize it to be 28x28 pixels.
> ~~~
> import cv2
> import matplotlib as plt
> digit = cv2.imread("digit.png")
> digit_gray = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
> digit_norm = digit_gray/255.0
> plt.matshow(digit_norm)
> plt.show()
> print("Your digit is",mlp.predict([digit_norm.reshape(784)]))
> ~~~
> {: .python}
{: .challenge}


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

The `sklearn.model_selection` module provides support for doing cross fold validation in scikit-learn. It can automatically partition our data for cross validation. 

Lets import this and cal it skl_msel `import sklearn.model_selection as skl_msel`

Now we can choose how many ways we'd like to split our data, three or four are common choices.

`kfold = skl_msel.KFold(4)`

Now we can loop through our data and test on each combination. The kfold.split function returns two variables and we'll have our for loop work through both of them. The train variable will contain a list of which items (by index number) we're currently using to train and the test one will contain the list of which items we're going to test on.

`for (train, test) in kfold.split(X):`

Now inside the loop we can select the data by doing `data_train = data[train]` and `labels_train = labels[train]`. This is a useful Python shorthand which will use the list of indicies from `train` to select which items from `data` and `labels` we use. We can repeat this process with the test set. 

~~~
    data_train = data[train]
    labels_train = labels[train]
    
    data_test = data[test]
    labels_test = labels[test]
 ~~~
 {: .python}
 
 
 Finally we need to train the classifier with the selected training data and then score it against the test data. The scores for each set of test data should be similar. 
 
 ~~~
    mlp.fit(data_train,labels_train)
    print("Testing set score", mlp.score(data_test, labels_test))
 ~~~
 {: .python}
 
 Once we've established that cross validation was ok we can go ahead and train using the entire dataset by doing `mlp.fit(data,labels)`.
 
 Here is the entire example program:

~~~
import matplotlib.pyplot as plt
import sklearn.datasets as skl_data
import sklearn.neural_network as skl_nn
import sklearn.model_selection as skl_msel

data, labels = skl_data.fetch_openml('mnist_784', version=1, return_X_y=True)
data = data / 255.0

mlp = skl_nn.MLPClassifier(hidden_layer_sizes=(50,), max_iter=50, random_state=1)

kfold = skl_msel.KFold(4)

for (train, test) in kfold.split(data):
    data_train = data[train]
    labels_train = labels[train]
    
    data_test = data[test]
    labels_test = labels[test]
    mlp.fit(data_train,labels_train)
    print("Training set score", mlp.score(data_train, labels_train))
    print("Testing set score", mlp.score(data_test, labels_test))
   
mlp.fit(data,labels)
~~~
{: .python}

## Deep Learning

Deep learning usually refers to newer neural network architectures which use a special type of network known as a convolutional network. Typically these have many layers and thousands of neurons. They're very good at tasks such as image recognition but take a long time to train and run. They're often used with GPU (Graphical Processing Units) which are good at executing multiple operations simultaneously. Its very common to use cloud computing or HPC systems with multiple GPUs attached. 

Scikit learn isn't really setup for Deep Learning and we'll have to rely on other libraries. Common choices include Google's TensorFlow, Keras, Torch or Darknet. There is however an interface layer between sklearn and tensorflow called skflow. A short example of doing this can be found at [https://www.kdnuggets.com/2016/02/scikit-flow-easy-deep-learning-tensorflow-scikit-learn.html](https://www.kdnuggets.com/2016/02/scikit-flow-easy-deep-learning-tensorflow-scikit-learn.html).

### Cloud APIs

Google, Microsoft, Amazon and many others now have Cloud based Application Programming Interfaces (APIs) where you can upload an image and have them return you the result. Most of these services rely on a large pre-trained (and often proprietary) neural network. 

> # Exercise: Try cloud image classification
> Take a photo with your phone camera or find an image online of a common daily scene. 
> Upload it Google's Vision AI example at https://cloud.google.com/vision/
> How many objects has it correctly classified? How many did it incorrectly classify?
> Try the same iamge with Microsoft's Computer Vision API at https://azure.microsoft.com/en-gb/services/cognitive-services/computer-vision/
> Does it do any better/worse than Google?
{: .challenge}



