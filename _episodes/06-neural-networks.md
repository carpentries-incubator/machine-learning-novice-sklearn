---
title: "Neural Networks"
teaching: 20
exercises: 30
questions:
- "How can we classify images using a neural network?"
objectives:
- "Explain the basic architecture of a perceptron."
- "Create a perceptron to encode a simple function."
- "Understand that a single perceptron cannot solve a problem requiring non-linear separability."
- "Understand that layers of perceptrons allow non-linear separable problems to be solved."
- "Train a multi-layer perceptron using scikit-learn."
- "Evaluate the accuracy of a multi-layer perceptron using real input data."
- "Understand that cross validation allows the entire data set to be used in the training process."
keypoints:
- "Perceptrons are artificial neurons which build neural networks."
- "A perceptron takes multiple inputs, multiplies each by a weight value and sums the weighted inputs. It then applies an activation function to the sum."
- "A single perceptron can solve simple functions which are linearly separable."
- "Multiple perceptrons can be combined to form a neural network which can solve functions that aren't linearly separable."
- "We can train a whole neural network with the back propagation algorithm. Scikit-learn includes an implementation of this algorithm."
- "Training a neural network requires some training data to show the network examples of what to learn."
- "To validate our training we split the the training data into a training set and a test set."
- "To ensure the whole dataset can be used in training and testing we can train multiple times with different subsets of the data acting as training/testing data. This is called cross validation."
- "Deep learning neural networks are a very powerful modern technique. Scikit learn does not support these but other libraries like Tensorflow do."
- "Several companies now offer cloud APIs where we can train neural networks on powerful computers."
---


# Introduction

Neural networks are a machine learning method inspired by how the human brain works. They are particularly good at doing pattern recognition and classification tasks, often using images as inputs. They are a well-established machine learning technique that has been around since the 1950s but have gone through several iterations since that have overcome fundamental limitations of the previous one. The current state-of-the-art neural networks is often referred to as deep learning.


## Perceptrons

Perceptrons are the building blocks of neural networks. They are an artificial version of a single neuron in the brain. They typically have one or more inputs and a single output. Each input will be multiplied by a weight and the value of all the weighted inputs are then summed together. Finally, the summed value is put through an activation function which decides if the neuron "fires" a signal. In some cases, this activation function is simply a threshold step function which outputs zero below a certain input and one above it. Other designs of neurons use other activation functions, but typically they have an output between zero and one and are still step-like in their nature.

![A diagram of a perceptron](../fig/perceptron.svg)

### Coding a perceptron

Below is an example of a perceptron written as a Python function. The function takes three parameters: Inputs is a list of input values, Weights is a list of weight values and Threshold is the activation threshold.

First let us multiply each input by the corresponding weight. To do this quickly and concisely, we will use the numpy multiply function which can multiply each item in a list by a corresponding item in another list.

We then take the sum of all the inputs multiplied by their weights. Finally, if this value is less than the activation threshold, we output zero, otherwise we output a one.

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
{: .language-python}


### Computing with a perceptron

A single perceptron can perform basic linear classification problems such as computing the logical AND, OR, and NOT functions.

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
{: .language-python}



AND:
~~~
inputs = [[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]]
for input in inputs:
    print(input,perceptron(input, [0.5,0.5], 1.0))
~~~
{: .language-python}



NOT:

The NOT function only has a single input but to make it work in the perceptron, we need to introduce a bias term which is always the same value. In this example, it is the second input. It has a weight of 1.0, the weight on the real input is -1.0.
~~~
inputs = [[0.0,1.0],[1.0,1.0]]
for input in inputs:
    print(input,perceptron(input, [-1.0,1.0], 1.0))
~~~
{: .language-python}

A perceptron can be trained to compute any function which has linear separability. A simple training algorithm called the perceptron learning algorithm can be used to do this and scikit-learn has its own implementation of it. We are going to skip over the perceptron learning algorithm and move straight onto more powerful techniques. If you want to learn more about it see [this page](https://computing.dcu.ie/~humphrys/Notes/Neural/single.neural.html) from Dublin City University.



### Perceptron limitations

A single perceptron cannot solve any function that is not linearly separable, meaning that we need to be able to divide the classes of inputs and outputs with a straight line. A common example of this is the XOR function shown below:

| Input 1 | Input 2 | Output |
| --------|---------|--------|
| 0       |0        |0       |
| 0       |1        |1       |
| 1       |0        |1       |
| 1       |1        |0       |

(Make a graph of this)

This function outputs a zero both when all its inputs are one or zero and its not possible to separate with a straight line. This is known as linear separability, when this limitation was discovered in the 1960s it effectively halted development of neural networks for over a decade in a period known as the "AI Winter".


## Multi-layer Perceptrons

A single perceptron cannot be used to solve a non-linearly separable function. For that, we need to use multiple perceptrons and typically multiple layers of perceptrons. They are formed of networks of artificial neurons which each take one or more inputs and typically have a single output. The neurons are connected together in large networks typically of 10s to 1000s of neurons. Typically, networks are connected in layers with an input layer, middle or hidden layer (or layers) and finally an output layer.

![A multi-layer perceptron](../fig/multilayer_perceptron.svg)

### Training Multi-layer perceptrons

Multi-layer perceptrons need to be trained by showing them a set of training data and measuring the error between the network's predicted output and the true value. Training takes an iterative approach that improves the network a little each time a new training example is presented. There are a number of training algorithms available for a neural network today, but we are going to use one of the best established and well known, the backpropagation algorithm. The algorithm is called back propagation because it takes the error calculated between an output of the network and the true value and takes it back through the network to update the weights. If you want to read more about back propagation, please see [this chapter](http://page.mi.fu-berlin.de/rojas/neural/chapter/K7.pdf) from the book "Neural Networks - A Systematic Introduction".

### Multi-layer perceptrons in scikit-learn

We are going to build a multi-layer perceptron for recognising handwriting from images. Scikit Learn includes some example handwriting data from the [MNIST data set](http://yann.lecun.com/exdb/mnist/), this consists of 70,000 images of hand written digits. Each image is 28x28 pixels in size (784 pixels in total) and is represented in grayscale with values between zero for fully black and 255 for fully white. This means we will need 784 perceptrons in our input layer, each taking the input of one pixel and 10 perceptrons in our output layer to represent each digit we might classify. If trained correctly then only the perceptron in the output layer to "fire" will be on the one representing the in the image (this is a massive oversimplification!).

We can import this dataset from `sklearn.datasets` with then load it into memory by calling the `fetch_openml` function.

~~~
import sklearn.datasets as skl_data
data, labels = skl_data.fetch_openml('mnist_784', version=1, return_X_y=True)
~~~
{: .language-python}

This creates two arrays of data, one called `data` which contains the image data and the other `labels` that contains the labels for those images which will tell us which digit is in the image. A common convention is to call the data `X` and the labels `y`

~~~
print(data.shape)
data.head()
~~~
{: .language-python}

As neural networks typically want to work with data that ranges between 0 and 1.0 we need to normalise our data to this range. Python has a shortcut which lets us divide the entire data array by 255 and store the result, we can simply do:
~~~
data = data / 255.0
~~~
{: .language-python}

Let us take 90% of the data for training and 10% for testing, so we will use the first 63,000 samples in the dataset as the training data and the last 7,000 as the test data. We can split these using a slice operator.

~~~
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.10, random_state=42, stratify=labels)
print(X_train.shape)
print(X_test.shape)
~~~
{: .language-python}

Now we need to initialise a neural network, scikit learn has an entire library `sklearn.neural_network` for this and the `MLPClassifier` class handles multi-layer perceptrons. This network takes a few parameters including the size of the hidden layer, the maximum number of training iterations we're going to allow, the exact algorithm to use, if we'd like verbose output about what the training is doing and the initial state of the random number generator.

In this example we specify a multi-layer perceptron with 50 hidden nodes, we allow a maximum of 50 iterations to train it, we turn on verbose output to see what's happening and initialise the random state to 1 so that we always get the same behaviour.

Now let us go ahead and train the network. This line will take about one minute to run. We do this by calling the `fit` function inside the `mlp` class instance. This needs two arguments the data itself and the labels showing what class each item should be classified to.

~~~
import sklearn.neural_network as skl_nn
mlp = skl_nn.MLPClassifier(hidden_layer_sizes=(50), max_iter=50, verbose=1, random_state=1)
mlp.fit(X_train,y_train)
~~~
{: .language-python}

Finally, let us score the accuracy of our network against both the original training data and the test data. If the training had converged to the point where each iteration of training was not improving the accuracy, then the accuracy of the training data should be 1.0 (100%). In scikit-learn, the score method for classifiers like MLPClassifier returns the accuracy of the classifier on the given test data and labels. Specifically, it computes the accuracy, which is the ratio of correctly classified samples to the total number of samples in the test dataset.

~~~
print("Training set score", mlp.score(X_train, y_train))
print("Testing set score", mlp.score(X_test, y_test))
~~~
{: .language-python}

### Prediction using a multi-layer perceptron

Now that we have trained a multi-layer perceptron, we can give it some input data and ask it to perform a prediction. In this case, our input data is a 28x28 pixel image, which can also be represented as a 784-element list of data. The output will be a number between 0 and 9 telling us which digit the network thinks we have supplied. The `predict` function in the `MLPClassifier` class can be used to make a prediction. Let us try using the first digit from our test set as an example.

Before we can pass it to the predictor, we have to extract one of the digits from the test set. We can use `iloc` on the dataframe to get hold of the first element in the test set. In order to present it to the predictor, we have to turn it into a numpy array which has the dimensions of 1x784 instead of 28x28. We can then call the `predict` function with this array as our parameter. This will return an array of predictions (as it could have been given multiple inputs), the first element of this will be the predicted digit. You may get a warning stating "X does not have valid feature names", this is because we didn't encode feature names into our X (digit images) data. We can now verify if the prediction is correct by looking at the corresponding item in the `y_test` array.

~~~
index = 0
test_digit = X_test.iloc[index].to_numpy().reshape(1,784)
test_digit_prediciton = mlp.predict(test_digit)[0]
print("Predicted value",test_digit_prediciton)
print("Actual value",y_test.iloc[index])
~~~
{: .language-python}


This should be the same value which is being predicted.

## Measuring Neural Network performance

We have now trained a neural network and tested prediction on a few images. This might have given us a feel for how well our network is performing, but it would be much more useful to have a more objective measure. Since recognising digits is a classification problem, we can measure how many predictions were correct in a set of test data. As we already have a test set of data with 7,000 images let us use that and see how many predictions the neural network has got right. We will loop through every image in the test set, run it through our predictor and compare the result with the label for that image. We will also keep a tally of how many images we got right and see what percentage were correct.

### Confusion Matrix
From earlier, know what percentage of images were correctly classified, but we don't know anything about the distribution of that across our different classes (the digits 0 to 9 in this case). A more powerful technique is known as a confusion matrix. Here we draw a grid with each class along both the x and y axis. The x axis is the actual number of items in each class and the y axis is the predicted number. In a perfect classifier there will be a diagonal line of values across the grid moving from the top left to bottom right corresponding to the number in each class and all other cells will be zero. If any cell outside of the diagonal is non-zero then it indicates a miss-classification. Scikit Learn has a function called `confusion_matrix` in the `sklearn.metrics` class which can display a confusion matrix for us. It will need two inputs, an array showing how many items were in each class for both the real data and the classifications. We already have the real data in the labels_test array, but we need to build it for the classifications by classifying each image (in the same order as the real data) and storing the result in another array.

~~~
from sklearn.metrics import confusion_matrix
import seaborn as sns

preds = []

for image in X_test.iterrows():
    # image contains a tuple of the row number and image data
    image = image[1].to_numpy().reshape(1,784)
    preds.append(mlp.predict(image)[0])

cm = confusion_matrix(y_test,preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
~~~
{: .language-python}

### Changing the learning rate hyperparameter
There are several hyperparameters which control the training of the data. One of these is called the learning rate, increasing this can reduce how many learning iterations we need. But make it too large and we will end up overshooting. Try tweaking this parameter by adding the parameter `learning_rate_init`, the default value of this is 0.001. Try increasing it to around 0.1.

~~~
mlp2 = skl_nn.MLPClassifier(hidden_layer_sizes=(50), max_iter=50, verbose=1, random_state=1, learning_rate_init=0.1)

mlp2.fit(X_train,y_train)

print("Training set score", mlp2.score(X_train, y_train))
print("Testing set score", mlp2.score(X_test, y_test))
~~~
{: .language-python}



### Using cross-validation to determing optimal hyperparameter value. 
Here, we use a train/test/validation split in order to assess how different values of the learning rate hyperparameter impact model performance.

The K-fold cross-validation procedure works as follows. We first split our data into training and test sets as usual. Then, or each possible value of the hyperparameter being tested...
1. We take our training data and split it into K equal-sized subsets (called folds).
2. We use K-1 of the folds for training the model, and the remaining fold for assessing the model's performance after training. This remaining fold is known as the validation set.
3. We repeat step 2 for each possible validation set (K total) and their associated training sets (the folds that were not left out)

After completing these steps, we can look at how the validation error varies with different values of the hyperparameter. We select the hyperparameter value / model that has the lowest validation error.

Finally, we train a model using all of the training data and hyperparameter value that was selected. Afterwards, we get a final assessment of the model's generalizeability by measuring its error on the test set. It is critical that the test set is not involved in the model selection process — otherwise you risk being overly optimistic about your model's ability to generalize.


~~~
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
import numpy as np

# Load your dataset and labels here (e.g., data and labels are loaded into X and y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.10, random_state=42, stratify=labels)

# Define a range of learning rate values to explore
learning_rates = [0.001, 0.01, 0.1]

# Initialize a dictionary to store cross-validation results
cv_results = {}

# Perform 4-fold cross-validation for each learning rate
for learning_rate in learning_rates:
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=3, learning_rate_init=learning_rate, random_state=42)
    scores = cross_val_score(mlp, X_train, y_train, cv=2, scoring='accuracy')
    cv_results[learning_rate] = scores.mean()

# Find the optimal learning rate based on cross-validation results
optimal_learning_rate = max(cv_results, key=cv_results.get)
optimal_score = cv_results[optimal_learning_rate]

print("Optimal Learning Rate:", optimal_learning_rate)
print("Optimal Cross-Validation Score:", optimal_score)

# Train the final model using the optimal learning rate on the combined train + validation set
final_X_train = np.vstack((X_train, X_validation))
final_y_train = np.hstack((y_train, y_validation))

final_model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, learning_rate_init=optimal_learning_rate, random_state=42)
final_model.fit(final_X_train, final_y_train)

# Assess the model on the held-out test set
test_score = final_model.score(X_test, y_test)
print("Test Set Score (Generalization Performance):", test_score)

~~~
{: .language-python}

### Optimizing multiple hyperparameters
If you want to optimize multiple hyperparameters at once, look into sklearn's GridSearchCV function. This will allow you to perform cross-validation across many hyperparameters at once. However, be careful not to go overboard in selecting too many possible hyperparameter combinations (esp. with complex models that take a while to train). It is typically better to investigate just a small subset of possible hyperparameter values so that you can find a good model (maybe not necessarily the BEST, but good enough) before the sun explodes.

## Deep Learning

Deep learning usually refers to newer neural network architectures which use a special type of network known as a convolutional network. Typically, these have many layers and thousands of neurons. They are very good at tasks such as image recognition but take a long time to train and run. They are often used with GPU (Graphical Processing Units) which are good at executing multiple operations simultaneously. It is very common to use cloud computing or HPC systems with multiple GPUs attached.

Scikit learn is not really setup for Deep Learning. We will have to rely on other libraries. Common choices include Google's TensorFlow, Keras, (Py)Torch or Darknet. There is however an interface layer between sklearn and tensorflow called skflow. A short example of doing this can be found at [https://www.kdnuggets.com/2016/02/scikit-flow-easy-deep-learning-tensorflow-scikit-learn.html](https://www.kdnuggets.com/2016/02/scikit-flow-easy-deep-learning-tensorflow-scikit-learn.html).

### Cloud APIs

Google, Microsoft, Amazon, and many others now have Cloud based Application Programming Interfaces (APIs) where you can upload an image and have them return you the result. Most of these services rely on a large pre-trained (and often proprietary) neural network.

> ## Exercise: Try cloud image classification
> Take a photo with your phone camera or find an image online of a common daily scene.
> Upload it Google's Vision AI example at https://cloud.google.com/vision/
> How many objects has it correctly classified? How many did it incorrectly classify?
> Try the same image with Microsoft's Computer Vision API at https://azure.microsoft.com/en-gb/services/cognitive-services/computer-vision/
> Does it do any better/worse than Google?
{: .challenge}

{% include links.md %}
