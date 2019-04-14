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


