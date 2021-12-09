---
title: "Ethics and Implications of Machine Learning"
teaching: 10
exercises: 5
questions:
- "What are the ethical implications of using machine learning in research?"
objectives:
- "To think about the ethical implications of machine learning."
- "To think about any ethical implications for using machine learning in research."
keypoints:
- "Machine learning is often thought of as unbiased and impartial. But if the training data is biased the machine learning will be."
- "Many machine learning algorithms can't explain how they arrived at a decision."
- "There is a lot of concern about how machine learning can be used for unethical purposes."
- "No machine learning system is 100% accurate, think about the implications of false positives and false negatives."
---

# Ethics and Machine Learning

There are increasing worries about the ethics of using machine learning.
In recent year's we've seen a number of worrying problems from machine learning entering all kinds of aspects of daily life and the economy:

* The first death from an autonomous car which failed to brake for a pedestrian.[\[1\]](https://www.forbes.com/sites/meriameberboucha/2018/05/28/uber-self-driving-car-crash-what-really-happened/)
* Highly targetted advertising based around social media and internet usage. [\[2\]](https://www.wired.com/story/big-tech-can-use-ai-to-extract-many-more-ad-dollars-from-our-clicks/)
* The outcomes of elections and referendums being influenced by highly targetted social media posts . This is compunded by the data being obtained without the users's consent. [\[3\]](https://www.vox.com/policy-and-politics/2018/3/23/17151916/facebook-cambridge-analytica-trump-diagram)
* The mass deploymeny of facial recognition technologies. [\[4\]](https://www.bbc.co.uk/news/technology-44089161)
* The possible first use of autonomous military robots making a decision to kill in battle. [\[5\]](https://www.theverge.com/2021/6/3/22462840/killer-robot-autonomous-drone-attack-libya-un-report-context)

## Problems with bias

Machine learning systems are often presented as more impartial and consistent ways to make decisions. For example sentencing criminals or
deciding if somebody should be granted bail. There have been a number of examples recently where machine learning systems have been shown to
be biased because the data they were trained on was already biased. This can occur due to the training data being unrepresentative and
under representing certain groups. For example if you were trying to automatically screen job candidates and used a sample of people the
same company had previously decided to employ then any biases in their past employment processes would be reflected in the machine learning.

## Problems with explaining decisions

Many machine learning systems (e.g. neural networks) can't really explain their decisions. Although the input and output are known trying to
explain why the training caused the network to behave in a certain way can be very difficult. If a decision is questioned by a human its
difficult to provide any rationale as to how a decision was arrived at.

## Problems with accuracy

No machine learning system is ever 100% accurate. Getting into the high 90s is usually considered good.
But when we're evaluating millions of data items this can translate into 100s of thousands of mis-identifications.
If the implications of these incorrect decisions are serious then it will cause major problems. For instance if it results in somebody
being imprisoned or even investigated for a crime or maybe just being denied insurance or a credit card.

## Energy Usage

Many machine learning systems (especially deep learning) need vast amounts of computational power which in turn can consume vast amounts of energy. Depending on the source of that energy this might account for significant amounts of fossil fuels being burned. It is not uncommon for a modern GPU accelerated computer to use several kilowatts of power, running this for one hour could easily use as much energy a typical home would use in an entire day. This can be particularly bad when models are constantly being retrained or when "parameter sweeps" are done to find the best set of parameters to train with.

# Ethics of machine learning in research

Not all research using machine learning will have major ethical implications.
Many research projects don't directly affect the lives of other people, but this isn't always the case.

Some questions you might want to ask yourself (and which an ethics committee might also ask you):

 * Will anything your machine learning system does make a decision that somehow affects a person's life?
 * Will anything your machine learning system does make a decision that somehow affects an animial's life?
 * Will you be using any people to create your training data? Will they have to look at any disturbing or traumatic material during the training process?
 * Are there any inherent biases in the dataset(s) you're using for training?
 * How much energy will this computation use? Are there more efficient ways to get the same answer?


> ## Exercise: Ethical implications of your own research
> Split into pairs or groups of three.
> Think of a use case for machine learning in your research areas.
> What ethical implications (if any) might there be from using machine learning in your research?
> Write down your group's answers in the etherpad.
{: .challenge}

{% include links.md %}
