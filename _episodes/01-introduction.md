---
title: "Introduction"
teaching: 20
exercises: 10
questions:
- "What is machine learning?"
- "What are some useful machine learning techniques?"
objectives:
- "Gain an overview of what machine learning is and the techniques available."
- "Understand how machine learning and artificial intelligence differ."
- "Be aware of some caveats when using Machine Learning."

keypoints:
- "Machine learning is a set of tools and techniques that use data to make predictions."
- "Artificial intelligence is a broader term that refers to making computers show human like intelligence."
- "Deep learning is a subset of machine learning."
- "All machine learning systems have limitations to be aware of."
---

# What is machine learning?

Machine learning is a set of techniques that enable computers to use data to improve in their performance of a given task. This is similar in concept to how humans learn to make predictions based upon previous experience and knowledge. Machine learning encompasses a wide range of activities, but broadly speaking it can be used to: find trends in a dataset, classify data into groups or categories, make decisions and predictions based upon data, and even "learn" how to interact with an environment when provided with goals to achieve.

### Machine learning in our daily lives

Machine learning has quickly become an important technology and is now frequently used to perform services we encounter in our daily lives. Here are just a few examples:

* Banks look for trends in transaction data to detect outliers that may be fraudulent
* Email inboxes use text to classify whether an email is spam or not, and adjust their classification rules based upon how we flag emails
* Travel apps use live and historic data to predict traffic, travel times, and journey routes
* Retail companies and streaming services use data to recommend new content we might like based upon our demographic and historical preferences
* Image, object, and pattern recognition is used to identify humans and vehicles, capture text, generate subtitles, and much more
* Self-driving cars and robots use object detection and performance feedback to improve their interaction with the world

> ## Categorise these examples into types of machine learning tasks
> Categorise the above examples of machine learning into one of the following four broad groupings:
>
> 1. finding trends in data
> 2. classifying data into groups and categories
> 3. making decisions and predictions
> 4. learning to interact in an environment
{: .challenge}

### Artificial Intelligence vs Machine Learning

The term machine learning (ML) is often mentioned alongside artificial intelligence (AI) and deep learning (DL). Deep learning is a subset of machine learning, and machine learning is a subset of artificial intelligence.

AI is a broad term used to describe a system possessing a "general intelligence" that can be applied to solve a diverse range problems, often mimicking the behaviour of intelligent biological systems. Another definition of AI dates back to the 1950s and Alan Turing's "Immitation Game". Turing said we could consider a system intelligent when it could fool a human into thinking they were talking to another human when they were actually talking to a computer. Modern attempts are getting close to fooling humans, but although there have been great advances in AI and ML research, human-like intelligence is only possible in a few specialist areas.

ML refers to techniques where a computer can "learn" patterns in data, usually by being shown many training examples. While ML-algorithms can learn to solve specific problems, or multiple similar problems, they are not considered to possess a general intelligence. ML-algorithms often need hundreds or thousands of examples to learn a task and are confined to tasks such as simple classifications. A human-like system could learn much quicker than this, and potentially learn from a single example by using it's knowledge of many other problems.

DL is a particular field of machine learning where algorithms called neural networks are used to create highly-complex systems. Large collections of neural networks are able to learn from vast quantities of data. Deep learning can be used to solve a wide range of problems, but it can also require huge amounts of input data and computational resources to train. 

The image below shows the relationships between artificial intelligence, machine learning and deep learning.

![An infographic showing some of the relationships between AI, ML, and DL](../fig/01_AI_ML_DL_differences.png)
The image above is by Tukijaaliwa, CC BY-SA 4.0, via Wikimedia Commons, original source

> ## Where have you encountered machine learning already?
> Now that we have explored machine learning in a bit more detail, discuss with the person next to you:
>
> 1. Where have I seen machine learning in use?
> 2. What kind of input data does that machine learning system use to make predictions/classifications?
> 3. Is there any evidence that your interaction with the system contributes to further training?
> 4. Do you have any examples of the system failing?
{: .challenge}

# What are some useful types of Machine Learning?

This lesson will introduce you to some of the key concepts and sub-domains of ML such as supervised learning, unsupervised learning, and neural networks.

The figure below provides a nice overview of some of the sub-domains of ML and the techniques used within each sub-domain. We recommend checking out the Scikit Learn [webpage](https://scikit-learn.org/stable/index.html) for additional examples of the topics we will cover in this lesson. We will cover topics highlighted in blue: classical learning techniques such as regression, classification, clustering, and dimension reduction, as well as a brief introduction to neural networks using perceptrons.

![Types of Machine Learning](../fig/ML_summary.png)
[Image from Vasily Zubarev via their blog](https://vas3k.com/blog/machine_learning/) with modifications in blue to denote lesson content.

# Limitations of machine learning

### Garbage in = garbage out

There is a classic expression in computer science, "garbage in = garbage out". This means that if the input data we use is garbage then the ouput will be too. If, for example, we try to use a machine learning system to find a link between two unlinked variables then it may well manage to produce a model attempting this, but the output will be meaningless. 

### Biases due to training data

The performance of a ML system depends on the breadth and quality of input data used to train it. If the input data contains biases or blind spots then these will be reflected in the ML system. For example, if we collect data on public transport use only from high-socioeconomic areas, the resulting input data may be biased due to the likelihood of people from those areas to use private transport vs public options.

### Extrapolation

We can only make reliable predictions about data which is in the same range as our training data. If we try to extrapolate beyond the boundaries of the training data we cannot be confident in our results. 

### Over fitting

Sometimes ML algorithms become over-trained and subsequently don't perform well when presented with real data. It's important to consider how many rounds of training a ML system has recieved and whether or not it may have become over-trained. 

### Inability to explain answers

Machine learning techniques will return an answer based on the input data and model parameters even if that answer is wrong. Most systems are unable to explain the logic used to arrive at that answer. This can make detecting and diagnosing problems difficult. 

{% include links.md %}
