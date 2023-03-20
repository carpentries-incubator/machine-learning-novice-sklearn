---
title: "Ethics and the Implications of Machine Learning"
teaching: 10
exercises: 5
questions:
- "What are the ethical implications of using machine learning in research?"
objectives:
- "Consider the ethical implications of machine learning in general and in research."
keypoints:
- "The results of machine learning reflect biases in the training and input data."
- "Many machine learning algorithms can't explain how they arrived at a decision."
- "Machine learning can be used for unethical purposes."
- "Consider the implications of false positives and false negatives."
---

# Ethics and machine learning

As machine learning has risen in visibility, so to have concerns around the ethics of using the technology to make predictions and decisions that will affect people in everyday life. For example:

* The first death from a driverless car which failed to brake for a pedestrian.[\[1\]](https://www.forbes.com/sites/meriameberboucha/2018/05/28/uber-self-driving-car-crash-what-really-happened/)
* Highly targetted advertising based around social media and internet usage. [\[2\]](https://www.wired.com/story/big-tech-can-use-ai-to-extract-many-more-ad-dollars-from-our-clicks/)
* The outcomes of elections and referenda being influenced by highly targetted social media posts. This is compunded by data being obtained without the users consent. [\[3\]](https://www.vox.com/policy-and-politics/2018/3/23/17151916/facebook-cambridge-analytica-trump-diagram)
* The widespread use of facial recognition technologies. [\[4\]](https://www.bbc.co.uk/news/technology-44089161)
* The potential for autonomous military robots to be deployed in combat. [\[5\]](https://www.theverge.com/2021/6/3/22462840/killer-robot-autonomous-drone-attack-libya-un-report-context)

## Problems with bias

Machine learning systems are often argued to be be fairer and more impartial in their decision-making than human beings, who are argued to be more emotional and biased, for example, when sentencing criminals or deciding if someone should be granted bail. But there are an increasing number of examples where machine learning systems have been exposed as biased due to the data they were trained on. This can occur due to the training data being unrepresentative or just under representing certain cases or groups. For example, if you were trying to automatically screen job candidates and your training data consisted only of people who were previously hired by the company, then any biases in employment processes would be reflected in the results of the machine learning.

## Problems with explaining decisions

Many machine learning systems (e.g. neural networks) can't really explain their decisions. Although the input and output are known, trying to
explain why the training caused the network to behave in a certain way can be very difficult. When decisions are questioned by a human it's
difficult to provide any rationale as to how a decision was arrived at.

## Problems with accuracy

No machine learning system is ever 100% accurate. Getting into the high 90s is usually considered good.
But when we're evaluating millions of data items this can translate into 100s of thousands of mis-identifications.
This would be an unacceptable margin of error if the results were going to have major implications for people, such as being imprisoned or structuring debt repayments.

## Energy use

Many machine learning systems (especially deep learning) need vast amounts of computational power which in turn can consume vast amounts of energy. Depending on the source of that energy this might account for significant amounts of fossil fuels being burned. It is not uncommon for a modern GPU-accelerated computer to use several kilowatts of power. Running this system for one hour could easily use as much energy a typical home in the OECD would use in an entire day. Energy use can be particularly high when models are constantly being retrained or when "parameter sweeps" are done to find the best set of parameters to train with.

# Ethics of machine learning in research

Not all research using machine learning will have major ethical implications.
Many research projects don't directly affect the lives of other people, but this isn't always the case.

Some questions you might want to ask yourself (and which an ethics committee might also ask you):

 * Will the results of your machine learning influence a decision that will have a significant effect on a person's life?
 * Will the results of your machine learning influence a decision that will have a significant effect on an animial's life?
 * Will you be using any people to create your training data, and if so, will they have to look at any disturbing or traumatic material during the training process?
 * Are there any inherent biases in the dataset(s) you're using for training?
 * How much energy will this computation use? Are there more efficient ways to get the same answer?


> ## Exercise: Ethical implications of your own research
> Split into pairs or groups of three.
> Think of a use case for machine learning in your research areas.
> What ethical implications (if any) might there be from using machine learning in your research?
> Write down your group's answers in the etherpad.
{: .challenge}

{% include links.md %}
