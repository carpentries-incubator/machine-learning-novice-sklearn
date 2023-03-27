---
title: "Classification"
teaching: 15
exercises: 20
questions:
- "How can I use scikit-learn to classify data?"
objectives:
- "Use two different methods to classify data"
- "Understand the difference between supervised and unsupervised learning"
- "Learn to validate and ?cross-validate? models"
[//]: # (- "Recall that scikit-learn has built in linear regression functions.")
keypoints:
- "Classification requires labelled data (is supervised)"
-
---

# Classification

Classification is the process of assigning items to classes, based on observation of some features. Where regression uses observations (x) to predict a numerical value (y), classification predicts a categorical fit to a class.

## Supervised vs. unsupervised learning
(this is probably introduced in Regression, so not needed?)

## The Penguin dataset
We're going to be using the penguins dataset, which comprises 342 observations of penguins of three different species: Adelie, Chinstrap & Gentoo. For each penguin we're given measurements of its bill length and depth (mm), flipper length (mm) and body mass (g).

source: {% https://github.com/allisonhorst/palmerpenguins %}

~~~
import seaborn as sns

dataset = sns.load_dataset('penguins')
dataset.head()
~~~
{: .language-python}

Our aim is to develop a classification model that will predict the species of a penguin given those measurements.

### Training-testing split
When undertaking any machine learning project, it's important to be able to evaluate how well your model works. In order to do this, we set aside some data (usually 20%) as a testing set, leaving the rest as your training dataset.

{callout} It's important to do this early, and to do all of your work with the training dataset - this avoids any risk of you as the developer introducing bias to the model based on your own observations of data in the testing set.

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

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
~~~
{: .language-python}

We'll use X_train and y_train to develop our model, and only look at X_test and y_test when it's time to evaluate its performance.

### Visualising the data
In order to better understand how a model might classify this data, we can first take a look at the data visually, to see what patterns we might identify.

~~~
fig01 = sns.scatterplot(X_train, x=feature_names[0], y=feature_names[1], hue=dataset['species'])
plt.show()
~~~
{: .language-python}

As there are four measurements for each penguin, we need a second plot to visualise all four dimensions:

~~~
fig23 = sns.scatterplot(X_train, x=feature_names[2], y=feature_names[3], hue=dataset['species'])
plt.show()
~~~
{: .language-python}

We can see that penguins from each species form fairly distinct spatial clusters in these plots, so that you could draw lines between those clusters to delineate each species. This is effectively what many classification algorithms do - using the training data to delineate the observation space (the 4 measurement dimensions) into classes. When given new observations, the model then finds which of those class areas that observation falls in to.

![Visualising the penguins dataset](../fig/e3_penguins_vis.png)

## Decision Tree
We'll first apply a decision tree classifier to the data. Decisions trees are conceptually similar to flow diagrams (or more precisely for the biologists: dichotomous keys) - they split the classification problem into a binary tree of comparisons, at each step comparing a measurement to a value, and moving left or right down the tree until a classification is reached.

(figure)

pros & cons

Training and using a decision tree in scikit-learn is straightforward:
~~~
from sklearn.tree import DecisionTreeClassifier, plot_tree

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

clf.predict(X_test)
~~~
{: .language-python}

We can conveniently check how our model did with the .score() function, which will make predictions and report what proportion of them were accurate:

~~~
clf.score(X_test, y_test)
~~~
{: .language-python}

We can also look at the decision tree that was generated:

~~~
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 10))
plot_tree(clf, class_names=class_names, feature_names=feature_names, filled=True, ax=fig.gca())
plt.show()
~~~
{: .language-python}

We can see from this that there's some very tortuous logic being used to tease out every single observation in the training set - for example the single purple Gentoo node at the bottom of the tree. If we truncated that branch to the second level (Chinstrap), we'd have a little inaccuracy, 5 non-Chinstraps in with 47 Chinstraps, but a less convoluted model. All of which is to say that, this model is clearly over-fit - it's developed a very complex delineation of the classification space in order to match every single observation, which will likely lead to poor results for new observations.

![Decision tree for classifying penguins](../fig/e3_decision_6_deep.png)

### Visualising the classification space
We can visualise the delineation produced, but only for two parameters at a time, so the model produced here isn't exactly that same as that used above:

~~~
from sklearn.inspection import DecisionBoundaryDisplay

f1 = feature_names[2]
f2 = feature_names[3]

clf = DecisionTreeClassifier()
clf.fit(X_train[[f1, f2]], y_train)

d = DecisionBoundaryDisplay.from_estimator(clf, X_train[[f1, f2]])

# labels = [class_names[i] for i in y_train]
sns.scatterplot(X_train, x=f1, y=f2, hue=y_train, palette='husl')
plt.show()
~~~
{: .language-python}

We can see that rather than clean lines between species, the decision tree produces orthogonal regions (as each decision only considers a single parameter). Again, we can see that the model is overfit - the decision space is far more complex than needed, with regions that only select a single point.

![Classification space for our decision tree](../fig/e3_decision_space.png)

## SVM
Next, we'll look at another commonly used classification algorithm, and see how it compares. Support Vector Machines (SVM) work in a way that is conceptually similar to your own intuition when first looking at the data - they devise a set of hyperplanes that delineate the parameter space, such that each region contains ideally only observations from one class, and the boundaries fall between classes.

### Normalising data
Unlike decision trees, SVMs require an additional pre-processing step for our data - we need it to be normalised. Our raw data has parameters with different magnitudes - bill length measured in 10's mm's vs. body mass measured in 1000's of grams. If we trained an SVM directly on this data, it would only consider the parameter with the greatest variance - body mass.

Normalising maps each parameter to a new range, so that it has a mean of 0, and a standard deviation of 1.

~~~
from sklearn import preprocessing

scalar = preprocessing.StandardScaler()
scalar.fit(X_train)
X_train_scaled = pd.DataFrame(scalar.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scalar.transform(X_test), columns=X_test.columns, index=X_test.index)
~~~
{: .language-python}

Note that we fit the scalar to our training data - we then use this same pre-trained scalar to transform our testing data.

With this scaled data, training the models works exactly the same as before.

~~~
from sklearn import svm

SVM = svm.SVC(kernel='poly', degree=3, C=1.5)
SVM.fit(X_train_scaled, y_train)

SVM.score(X_test_scaled, y_test)
~~~
{: .language-python}

We can again visualise the decision space produced, also using only two parameters:

~~~
x2 = X_train_scaled[[feature_names[0], feature_names[1]]]

SVM = svm.SVC(kernel='poly', degree=3, C=1.5)
SVM.fit(x2, y_train)

DecisionBoundaryDisplay.from_estimator(SVM, x2) #, ax=ax)
sns.scatterplot(x2, x=feature_names[0], y=feature_names[1], hue=dataset['species'])
plt.show()
~~~
{: .language-python}

While this SVM model performs slightly worse than our decision tree (95.6% vs. 97.1%), we can see that the decision space is much simpler, and less likely to be overfit to the data.

![Classification space generated by the SVM model](../fig/e3_svc_decision_space.png)

## Reducing over-fitting in the decision tree
We can reduce the over-fitting of our decision tree model by limiting its depth, forcing it to use less decisions to produce a classification, and resulting in a simpler decision space.

~~~
max_depths = [1, 2, 3, 4, 5]

accuracy = []
for i, d in enumerate(max_depths):
    clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    accuracy.append((d, acc))

acc_df = pd.DataFrame(accuracy, columns=['depth', 'accuracy'])

sns.lineplot(acc_df, x='depth', y='accuracy')
plt.xlabel('Tree depth')
plt.ylabel('Accuracy')
~~~
{: .language-python}

Here we can see that a maximum depth of two performs just as well as our original model with a depth of five - in this example if even performs a little better.

![Performance of decision trees of various depths](../fig/e3_dt_overfit.png)

Reusing our visualisation code from above, we can inspect our simplified decision tree and decision space:

~~~
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X_train, y_train)

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
clf.fit(X_train[[f1, f2]], y_train)

d = DecisionBoundaryDisplay.from_estimator(clf, X_train[[f1, f2]])

sns.scatterplot(X_train, x=f1, y=f2, hue=y_train, palette='husl')
plt.show()
~~~
{: .language-python}

We can see that both the tree and the decision space are much simpler, but still do a good job of classifying our data. We've succeeded in reducing over-fitting.

![Classification space of the simplified decision tree](../fig/e3_decision_space_depth_2.png)

{callout box thing} 'Max Depth' is an example of a *hyper-parameter* to the decision tree model. Where models use the parameters of an observation to predict a result, hyper-parameters are used to tune how a model works. Each model you encounter will have its own set of hyper-parameters, each of which affects model behaviour and performance in a different way. The process of adjusting hyper-parameters in order to improve model performance is called hyper-parameter tuning.


# September
### Note that care is needed when splitting data
- You generally want to ensure that each class is represented proportionately in both training + testing (beware just taking the first 80%)
- Sometimes you want to make sure a group is excluded from the train/test split, e.g.: when multiple samples come from one individual
- This is often called stratification
See https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators
