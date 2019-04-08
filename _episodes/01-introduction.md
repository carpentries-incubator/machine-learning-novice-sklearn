---
title: "Introduction"
teaching: 0
exercises: 0
questions:
- What is machine learning?"
objectives:
- "Gain an overview of what machine learning is."
- "Understand how machine learning and artificial intelligence differ."
- "Understand some common examples of machine learning being used in our daily lives"
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---
FIXME


# What is machine learning?

Machine learning is a set of of tools and techniques which let us find patterns in data. 

(FIXME): introduce both prediction and classification here? 


In its most basic form machine learning lets us predict the value of something given at least one other variable. For example given a country's GDP we can predict what its life expectancy might be. We can do this by using data from lots of other countries to form a model linking the two variables. We can now use the model to calculate what the life expectancy of a given country will be if we know its GDP. If our machine learning system works correctly it will hopefully give us an accurate answer. 

## Training Data

Machine learning systems "learn" by taking a series of input data and output data and using it to form a model. The maths behind the machine learning doesn't care what the data is as long as it can represented numerically or categorised. Some examples might include:

* predicting a person's weight based on their height
* predicting commute times given traffic conditions
* predicting house prices given stock market prices


Typically we will need to train our models with hundreds, thousands or even millions of examples before they work well enough to do any useful predictions on. 

### Types of output

Some systems will involve a continous scale of outputs, such as predicting the price of something. But many machine learning systems predict which class something belongs to given an input value. For example a system to recognise hand writing from an input image will need to classify the output into one of a set of potential characters. 


## Machine learning vs Artificial Intelligence

A lot of hype around machine learning right now. Recent advances in computer hardware and machine learning algorithms have made it a lot more useful, but its been around over 50 years. 

Artificial Intelligence often means a system with general intelligence, able to solve any problem. AI is a very broad term. 

ML trained to work on a particular problem. Can appear to "learn" but isn't a general intelligence. Often needs many many examples to learn. A human like system could learn from a single example. 

Turing Test 



# Examples of machine learning

## Applications of machine learning

### Machine learning in our daily lives

Speech to text

Siri/Alexa/Google assistant

https://www.youtube.com/watch?v=J3lYLphzAnw (turn on subtitles)

Image recognition

https://www.youtube.com/watch?v=eve8DkkVdhI

object classification

https://www.youtube.com/watch?v=VOC3huqHrss

character recognition

https://www.youtube.com/watch?v=ocB8uDYXtt0


### Example of machine learning in research

Classifying remote sensing images to find water.

https://pure.aber.ac.uk/portal/en/publications/automatic-detection-of-open-and-vegetated-water-bodies-using-sentinel-1-to-map-african-malaria-vector-mosquito-breeding-habitats(be685278-6eb8-46d0-aad9-be17add5639a).html

https://pure.aber.ac.uk/portal/files/29140808/remotesensing_11_00593.pdf


looking for breast cancer in medical images

https://pure.aber.ac.uk/portal/en/publications/automated-breast-ultrasound-lesions-detection-using-convolutional-neural-networks(6ede5cbd-75d6-48d6-9f7c-cb943081bbb0).html

https://pure.aber.ac.uk/portal/files/28421096/08003418.pdf

predicting what cows are doing from GPS data

https://pure.aber.ac.uk/portal/en/publications/a-novel-behavioral-model-of-the-pasturebased-dairy-cow-from-gps-data-using-data-mining-and-machine-learning-techniques(2660287d-f99c-428d-8a24-194a152dcccb).html



### Limitations of Machine Learning

#### Garbage In = Garbage Out

There is a classic expression in Computer Science, "Garbage In = Garbage Out". This means that if the input data we use is garbage then the ouput will be too. If for instance we try to get a machine learning system to find a link between two unlinked variables then it might still come up with a model that attempts this, but the output will be meaningless. 

#### Extrapolation

We can only make reliable predictions about data which is in the same range as our training data. If we try to extrapolate beyond what was covered in the training data we'll probably get wrong answers. 

#### Over fitting

Some ML techniques more resistant to this than others 



## Classification







# What is machine learning?

data input

training

classification








> ## Where have you encountered machine learning already?
>
> Discuss with the person next to you:
>
> 1. Where have I seen machine learning in use?
> 2. What kind of input data does that machine learning system use?
> 3. Is there any evidence that your interaction with the system contributes to further training?
> 4. Do you have any examples of the system failing?
>
> Write your answers into the etherpad.
{: .challenge}

