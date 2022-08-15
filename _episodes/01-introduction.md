---
title: "Introduction"
teaching: 30
exercises: 10
questions:
- What is machine learning?
objectives:
- "Gain an overview of what machine learning is."
- "Understand how machine learning and artificial intelligence differ."
- "Understand some common examples of machine learning being used in our daily lives"
keypoints:
- "Machine learning is a set of tools and techniques to find patterns in data."
- "Some machine learning techniques are useful for predicting something given some input data."
- "Some machine learning techniques are useful for classifying input data and working out which class it belongs to."
- "Artificial Intelligence is a broader term that refers to making computers show human like intelligence."
- "Some people say Artificial Intelligence to mean machine learning"
- "All machine learning systems have some kinds of limitations"
---

# What is machine learning?

Machine learning is a set of of tools and techniques which let us find patterns in data. This lesson will introduce you to a few of these techniques, but there are many more which we simply don't have time to cover here. 

The techniques breakdown into two broad categories, predictors and classifiers. Predictors are used to predict a value (or set of value) given a set of inputs, for example trying to predict the cost of something given the economic conditions and the cost of raw materials or predicting a country's GDP given its life expectancy. Classifiers try to classify data into different categories, for example deciding what characters are visible in a picture of some writing or if a message is spam or not. 


## Training Data

Many (but not all) machine learning systems "learn" by taking a series of input data and output data and using it to form a model. The maths behind the machine learning doesn't care what the data is as long as it can represented numerically or categorised. Some examples might include:

* predicting a person's weight based on their height
* predicting commute times given traffic conditions
* predicting house prices given stock market prices
* classifying if an email is spam or not
* classifying what if an image contains a person or not


Typically we will need to train our models with hundreds, thousands or even millions of examples before they work well enough to do any useful predictions or classifications with them. 

Some systems will do training as a one shot process which produces a model. Others might try to continuosly refine their training through the real use of the system and human feedback to it. For example every time you mark an email as spam or not spam you are probably contributing to further training of your spam filter's model. 

### Types of output

Predictors will usually involve a continuos scale of outputs, such as the price of something. Classifiers will tell you which class (or classes) are present in the data. For example a system to recognise hand writing from an input image will need to classify the output into one of a set of potential characters. 


## Machine learning vs Artificial Intelligence

Artificial Intelligence often means a system with general intelligence, able to solve any problem. AI is a very broad term. ML systems are usually trained to work on a particular problem. But they can appear to "learn" but isn't a general intelligence that can solve anything a human could. They often need hundreds or thousands of examples to learn and are confined to relatively simple classifications. A human like system could learn from a single example. 

Another definition of Artificial Intelligence dates back to the 1950s and Alan Turing's "Immitation Game". This said that we could consider a system intelligent when it could fool a human into thinking they were talking to another human when they were actually talking to a computer. Modern attempts at this are getting close to fooling humans, but we are still a very long way from a machine which has full human like intelligence.

### Over Hyping of Artificial Intelligence and Machine Learning

There is a lot of hype around machine learning and artificial intelligence right now, while many real advances have been made a lot of people are overstating what can be achieved. Recent advances in computer hardware and machine learning algorithms have made it a lot more useful, but its been around over 50 years. 

The [Gartner Hype Cycle](https://www.gartner.com/en/research/methodologies/gartner-hype-cycle) looks at which technologies are being over-hyped. In the August 2018 analysis AI Platform as a service, Deep Learning chips, Deep learning neural networks, Conversational AI and Self Driving Cars are all shown near the "Peak of inflated expectations". 

![The Gartner Hype Cycle curve](https://upload.wikimedia.org/wikipedia/commons/9/94/Gartner_Hype_Cycle.svg)
[Image from Jeremy Kemp via Wikimedia](https://en.wikipedia.org/wiki/File:Gartner_Hype_Cycle.svg)

# Applications of machine learning

## Machine learning in our daily lives

 * [Image recognition](https://www.youtube.com/watch?v=eve8DkkVdhI)
 * [Object classification](https://www.youtube.com/watch?v=VOC3huqHrss)
 * [Character recognition](https://www.youtube.com/watch?v=ocB8uDYXtt0)
 * [Insurance payout predictions](https://www.youtube.com/watch?v=Q3vknDOy6Bs)
 * [Crime prediction](https://www.youtube.com/watch?v=7Ly7yAzLDjA)


## Example of machine learning in research
 * [Classifying remote sensing images to find water.](https://pure.aber.ac.uk/portal/files/29140808/remotesensing_11_00593.pdf)
 * [Looking for breast cancer in medical images](https://pure.aber.ac.uk/portal/files/28421096/08003418.pdf)
 * [Predicting what cows are doing from GPS data](https://pure.aber.ac.uk/portal/files/6707587/JDS_DairyModel_Revised_2.docx)



# Limitations of Machine Learning

## Garbage In = Garbage Out

There is a classic expression in Computer Science, "Garbage In = Garbage Out". This means that if the input data we use is garbage then the ouput will be too. If for instance we try to get a machine learning system to find a link between two unlinked variables then it might still come up with a model that attempts this, but the output will be meaningless. 

## Bias or lacking training data

Input data may also be lacking enough diversity to cover all examples. Due to how the data was obtained there might be biases in it that are then reflected in the ML system. For example if we collect data on crime reporting it could be biased towards wealthier areas where crimes are more likely to be reported. Historical data might not cover enough history.

## Extrapolation

We can only make reliable predictions about data which is in the same range as our training data. If we try to extrapolate beyond what was covered in the training data we'll probably get wrong answers. 

## Over fitting

Sometimes ML algorithms become over trained to their training data and struggle to work when presented with real data. In some cases it best not to train too many times. 

## Inability to explain answers

Many machine learning techniques will give us an answer given some input data even if that answer is wrong. Most are unable to explain any kind of logic in arriving at that answer. This can make diagnosing and even detecting problems with them difficult. 

> ## Where have you encountered machine learning already?
>
> Discuss with the person next to you:
>
> 1. Where have I seen machine learning in use?
> 2. What kind of input data does that machine learning system use to make predictions/classifications?
> 3. Is there any evidence that your interaction with the system contributes to further training?
> 4. Do you have any examples of the system failing?
>
> Write your answers into the etherpad.
{: .challenge}

{% include links.md %}
