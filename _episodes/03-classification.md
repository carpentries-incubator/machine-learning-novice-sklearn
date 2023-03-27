---
title: "Classification"
teaching: 15
exercises: 20
questions:
- "How can I classify data into known categories?"
objectives:
- "Use two different supervised methods to classify data."
- "Learn about the concept of hyper-parameters."
- "Learn to validate and ?cross-validate? models"
keypoints:
- "Classification requires labelled data (is supervised)"
---

# Classification

Classification is a supervised method to recognise and group data objects into a pre-determined categories. Where regression uses labelled observations to predict a continuous numerical value, classification predicts a discrete categorical fit to a class. Classification in ML leverages a wide range of algorithms to classify a set of data/datasets into their respective categories.

In this lesson we are going to introduce the concept of supervised classification by classifying penguin data into different species of penguins using Scikit-Learn.

### The penguins dataset
We're going to be using the penguins dataset of Allison Horst, published [here](https://github.com/allisonhorst/palmerpenguins) in 2020, which is comprised of 342 observations of three species of penguins: Adelie, Chinstrap & Gentoo. For each penguin we have measurements of bill length and depth (mm), flipper length (mm), body mass (g), and information on species, island, and sex.

~~~
import seaborn as sns

dataset = sns.load_dataset('penguins')
dataset.head()
~~~
{: .language-python}

Our aim is to develop a classification model that will predict the species of a penguin based upon measurements of those variables.

As a rule of thumb for ML/DL modelling, it is best to start with a simple model and progressively add complexity in order to meet our desired classification performance.

For this lesson we will limit our dataset to only numerical values such as bill_length, bill_depth, flipper_length, and body_mass while we attempt to classify species.

The above table contains multiple categorical objects such as species. If we attempt to include the other categorical fields, island and sex, we hinder classification performance due to the complexity of the data.

### Training-testing split
When undertaking any machine learning project, it's important to be able to evaluate how well your model works. In order to do this, we set aside some data (usually 20%) as a testing set, leaving the rest as your training dataset.

> ## Why do we do this?
> It's important to do this early, and to do all of your work with the training dataset - this avoids any risk of you introducing bias to the model based on your own observations of data in the testing set, and can highlight when you are over-fitting on your training data.
{: .callout}

~~~
# Extract the data we need
feature_names = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
dataset.dropna(subset=feature_names, inplace=True)

class_names = dataset['species'].unique()

X = dataset[feature_names]

Y = dataset['species']
~~~
{: .language-python}

Having extracted our features (X) and labels (Y), we can now split the data

~~~
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
~~~
{: .language-python}

We'll use x_train and y_train to develop our model, and only look at x_test and y_test when it's time to evaluate its performance.

### Visualising the data
In order to better understand how a model might classify this data, we can first take a look at the data visually, to see what patterns we might identify.

~~~
import matplotlib.pyplot as plt

fig01 = sns.scatterplot(x_train, x=feature_names[0], y=feature_names[1], hue=dataset['species'])
plt.show()
~~~
{: .language-python}

As there are four measurements for each penguin, we need quite a few plots to visualise all four dimensions against each other. Here is a handy Seaborn function to do so:

~~~
sns.pairplot(dataset, hue="species")
plt.show()
~~~
{: .language-python}

We can see that penguins from each species form fairly distinct spatial clusters in these plots, so that you could draw lines between those clusters to delineate each species. This is effectively what many classification algorithms do. They use the training data to delineate the observation space, in this case the 4 measurement dimensions, into classes. When given a new observation, the model finds which of those class areas the new observation falls in to.

![Visualising the penguins dataset](../fig/e3_penguins_vis.png)

## Classification using a decision tree
We'll first apply a decision tree classifier to the data. Decisions trees are conceptually similar to flow diagrams (or more precisely for the biologists: dichotomous keys). They split the classification problem into a binary tree of comparisons, at each step comparing a measurement to a value, and moving left or right down the tree until a classification is reached.

(figure)

Training and using a decision tree in Scikit-Learn is straightforward:
~~~
from sklearn.tree import DecisionTreeClassifier, plot_tree

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

clf.predict(x_test)
~~~
{: .language-python}

We can conveniently check how our model did with the .score() function, which will make predictions and report what proportion of them were accurate:

~~~
clf_score = clf.score(x_test, y_test)
~~~
{: .language-python}

We can also look at the decision tree that was generated:

~~~
fig = plt.figure(figsize=(12, 10))
plot_tree(clf, class_names=class_names, feature_names=feature_names, filled=True, ax=fig.gca())
plt.show()
~~~
{: .language-python}

We can see from this that there's some very tortuous logic being used to tease out every single observation in the training set. For example, the single purple Gentoo node at the bottom of the tree. If we truncated that branch to the second level (Chinstrap), we'd have a little inaccuracy, 5 non-Chinstraps in with 47 Chinstraps, but a less convoluted model.

The tortuous logic, such as the bottom purple Gentoo node, is a clear indication that this model has been over-fitted. It has developed a very complex delineation of the classification space in order to match every single observation, which will likely lead to poor results for new observations.

![Decision tree for classifying penguins](../fig/e3_dt_6.png)

### Visualising the classification space
We can visualise the delineation produced, but only for two parameters at a time, so the model produced here isn't exactly the same as that used above:

~~~
from sklearn.inspection import DecisionBoundaryDisplay

f1 = feature_names[2]
f2 = feature_names[3]

clf = DecisionTreeClassifier()
clf.fit(x_train[[f1, f2]], y_train)

d = DecisionBoundaryDisplay.from_estimator(clf, x_train[[f1, f2]])

# labels = [class_names[i] for i in y_train]
sns.scatterplot(x_train, x=f1, y=f2, hue=y_train, palette='husl')
plt.show()
~~~
{: .language-python}

We can see that rather than clean lines between species, the decision tree produces orthogonal regions as each decision only considers a single parameter. Again, we can see that the model is over-fitting as the decision space is far more complex than needed, with regions that only select a single point.

![Classification space for our decision tree](../fig/e3_dt_space_6.png)

## Classification using support vector machines
Next, we'll look at another commonly used classification algorithm, and see how it compares. Support Vector Machines (SVM) work in a way that is conceptually similar to your own intuition when first looking at the data. They devise a set of hyperplanes that delineate the parameter space, such that each region contains ideally only observations from one class, and the boundaries fall between classes.

### Normalising data
Unlike decision trees, SVMs require an additional pre-processing step for our data. We need to normalise it. Our raw data has parameters with different magnitudes such as bill length measured in 10's of mm's, whereas body mass is measured in 1000's of grams. If we trained an SVM directly on this data, it would only consider the parameter with the greatest variance (body mass).

Normalising maps each parameter to a new range so that it has a mean of 0 and a standard deviation of 1.

~~~
from sklearn import preprocessing
import pandas as pd

scalar = preprocessing.StandardScaler()
scalar.fit(x_train)
x_train_scaled = pd.DataFrame(scalar.transform(x_train), columns=x_train.columns, index=x_train.index)
x_test_scaled = pd.DataFrame(scalar.transform(x_test), columns=x_test.columns, index=x_test.index)
~~~
{: .language-python}

Note that we fit the scalar to our training data - we then use this same pre-trained scalar to transform our testing data.

With this scaled data, training the models works exactly the same as before.

~~~
from sklearn import svm

SVM = svm.SVC(kernel='poly', degree=3, C=1.5)
SVM.fit(x_train_scaled, y_train)

svm_score = SVM.score(x_test_scaled, y_test)
print("Decision tree score is ", clf_score)
print("SVM score is ", svm_score)
~~~
{: .language-python}

We can again visualise the decision space produced, also using only two parameters:

~~~
x2 = x_train_scaled[[feature_names[0], feature_names[1]]]

SVM = svm.SVC(kernel='poly', degree=3, C=1.5)
SVM.fit(x2, y_train)

DecisionBoundaryDisplay.from_estimator(SVM, x2) #, ax=ax
sns.scatterplot(x2, x=feature_names[0], y=feature_names[1], hue=dataset['species'])
plt.show()
~~~
{: .language-python}

While this SVM model performs slightly worse than our decision tree (95.6% vs. 97.1%), we can see that the decision space is much simpler, and less likely to be overfit to the data.

![Classification space generated by the SVM model](../fig/e3_svc_space.png)

## Reducing over-fitting in the decision tree
We can reduce the over-fitting of our decision tree model by limiting its depth, forcing it to use less decisions to produce a classification, and resulting in a simpler decision space.

~~~
max_depths = [1, 2, 3, 4, 5]

accuracy = []
for i, d in enumerate(max_depths):
    clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)

    accuracy.append((d, acc))

acc_df = pd.DataFrame(accuracy, columns=['depth', 'accuracy'])

sns.lineplot(acc_df, x='depth', y='accuracy')
plt.xlabel('Tree depth')
plt.ylabel('Accuracy')
plt.show()
~~~
{: .language-python}

Here we can see that a maximum depth of two performs just as well as our original model with a depth of five. In this example it even performs a little better.

![Performance of decision trees of various depths](../fig/e3_dt_overfit.png)

Reusing our visualisation code from above, we can inspect our simplified decision tree and decision space:

~~~
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(x_train, y_train)

fig = plt.figure(figsize=(12, 10))
plot_tree(clf, class_names=class_names, feature_names=feature_names, filled=True, ax=fig.gca())
plt.show()
~~~
{: .language-python}

Noting the added max_depth=2 parameter.

~~~
f1 = feature_names[2]
f2 = feature_names[3]

clf = DecisionTreeClassifier(max_depth=2)
clf.fit(x_train[[f1, f2]], y_train)

d = DecisionBoundaryDisplay.from_estimator(clf, x_train[[f1, f2]])

sns.scatterplot(x_train, x=f1, y=f2, hue=y_train, palette='husl')
plt.show()
~~~
{: .language-python}

We can see that both the tree and the decision space are much simpler, but still do a good job of classifying our data. We've succeeded in reducing over-fitting.

![Simplified decision tree](../fig/e3_dt_2.png)

![Classification space of the simplified decision tree](../fig/e3_dt_space_2.png)

> ## Hyper-parameters: parameters that tune a model
> 'Max Depth' is an example of a *hyper-parameter* for the decision tree model. Where models use the parameters of an observation to predict a result, hyper-parameters are used to tune how a model works. Each model you encounter will have its own set of hyper-parameters, each of which affects model behaviour and performance in a different way. The process of adjusting hyper-parameters in order to improve model performance is called hyper-parameter tuning.
{: .callout}


### Note that care is needed when splitting data
- You generally want to ensure that each class is represented proportionately in both training and testing (beware of just taking the first 80%).
- Sometimes you want to make sure a group is excluded from the train/test split, e.g.: when multiple samples come from one individual.
- This is often called stratification
See [Scikit-Learn](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators) for more information.
