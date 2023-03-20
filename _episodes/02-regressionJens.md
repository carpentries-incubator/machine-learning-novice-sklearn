---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

Thoughts to be included longterm
- some of the lengthy python code could be condensed into functions that we define and then call
- maybe even put an interactive slider (scale 1-10) to select the degree of the polynomial
- plot might be saved and reloaded as sub-plot?
  - fig, ax = plt.subplots() is more concise than this: fig = plt.figure() ax = fig.add_subplot(111)


# Step 0: Setting the Scene
- in this episode, we will apply one of *the two big ML approaches*: Regression and classifiction (supervised)/clustering (unsupervised) which we will talk about in the next episodes
- imagine in a university-setting, you are frequently asked how hard an exam is, how much time to invest for its preparation and ultimately what restult to expect 
- For this purpose, we fundamentally have to do two things: 
  - Task 1: Create a hypothesis about what input we need to predit the exam result
  - Task 2: Get data, build a *good* model to achieve our end goal of predicting quantitative values (here: The exam results, other similar examples are continuous values such as the price of a car, the weight of a dog).
- We guess that a direct corellation exists between the hours spent studying for the exam and the result. That is our simplification/abstraction/hypothesis. 
- while this simplified problem can be solved purely with hand-calculations (maths, statics), we want to establish a link to ML (best) practices (and continue doing so in the long run). This means that:
  - We will start with a clean dataset. That is unusual in practice. Often we have to consolidate several inputs into one neat table, for example. Some argue that this is of of the most time-consuming aspect while working with data. 
  - We will split an existing dataset into a training and a testing dataset. Remember, we do not follow classic analytic approaches where we define step by step what the algorithm has to do, we still define the basics (also called *hyper-parameters*), but in a true ML-fashion, the model is trained on data samples (where the *parameters*) are found and in a next step, the success (some refer to this as *accuracy* or *validity*) is evaluated. 
  - Though the maths/statics background is interesting, we will focus on using ML libraries such as Scikit Learn (sklearn) and will use predefined functions instead of creating these from scratch.


# Step 1: Envisioned Model
- For Task 1, we form the hypothesis that we can predict exam results based on the hours of studying invested. Upon reflection we can imagine that there might be more to that; previous experience, procastinating while stuyding, etc. etc.  
- To create our dataset, we have collected some historic data by asking former students for the time spent preparing the exam and their results.
- Let's firstly get an impression of our collected data. Maybe we identify some relationship (a trend, for exmaple) which will help us to come up with a prediction (such as: *you are likely to get this many points in the exam* based on the hours you plan to study for it)
- As an outlook, after we got an understanding of the data, we might find a simplification that we find most appropriate based on some of our data (the training data) and then gauge how well our prediction model performs by now feeding the testing data in, letting the algorithm perform its task and then compare predicted vs. actual values. 



# Step 2: Get data and visualise it
Lets get some training data and visualise it

```python
%matplotlib inline
import matplotlib.pyplot as plt # good practice to import libraries firstly

# The results from our data collection
x_data = [2,3,5,7,7.2,9] #hours of learning
y_data = [4,5,7,10,13,15] #test results

# The visualiation starts here
plt.scatter(x_data, y_data, c='k')
plt.xlabel('hours of learning')
plt.ylabel('test results 0 to 15, 15 being best')
plt.title('Scatter plot of all our collected data')
plt.show()
```

# Step 3: Split dataset into Training and Testing data 

```python
import numpy as np
from sklearn.model_selection import train_test_split # again, we use common ML libraries (yes, this task could be performed manually)

# we could write x_train, x_test, y_train, y_test but as we will mostly work with the training data for now, we just call x_train --> x and y_train --> y 
x, x_test, y, y_test = train_test_split( 
x_data, y_data, test_size=0.25, random_state=42)
print(f"The x-values used to train our model on are: {x}") # Note the random sequence which is also good ML practice
```

# Step 4: Create a model
Let's have another look at the graph we created a few lines above. It seems tempting to simplify (in other words, to *approximate* or to *model*) the given points (also referred to as a *point cloud*) by one line. This is also referred to as *linear regression*. 

Let's call this approximation by a carefully positioned line **Use Case 1** or **UC1** for short. Referring to maths, suitable values for $m$ and $b$ would have to be found for such a line given by an equation of $y=mx +c$.

Another similar approach is called *polynomial regression* which means not a line but a curve will be fit. This concept has many (*poly*) names (*nom*) or in our case: *terms*. How many terms is determined by something called the *degree*. For a degree two (which describes the highest power or exponent in the related equation) we get $y=ax^2 + bx +c$.  

Therefore, the *linear regression* can be consdiered one special case of polynomial regression with a degree of one. So, still $y=mx +c$. (A degree of five would entail and equation of $y=ax^5 +bx^4 \dots$). We will get back to polynomial regression soon. 


# Step 5: Code UC1 in Python using SciKitLearn
- 5.1 Create a linear regression model in Python
- 5.2 Revisit the resulting equations and the plots, including the deviation as dotted lines
- 5.3 Create a table with relevant statistical metrics
  - 5.3.1 residuals (list)
  - 5.3.2 sum of residuals; highlighting how different signs might cancel each other out
  - 5.3.3 workaround to square the delta and root; or Python is in part (but not only) a fancy calculator, so just use the abs() function
  - 5.3.4 R-squared, exaplain, put more info from Youtube StatQuest


```python
# Use Case 1 - Linear Regression 
# Calculate the fit line with degree 1 which means a LINEAR REGRESSION
Model_UC1 = np.polyfit(x, y, deg=1) # this is the regession and we specify the degree to be 1
Equation_UC1 = np.poly1d(Model_UC1) # poly1d helps us humans read the equation and makes several aspects callable from other functions

print(f"The equation for UC1 is y = {Equation_UC1}") # We use f-notation which is a handy shortcut to print text and variables
# the alternative would be 
# print('The euqation for UC1 is y =' +str(Equation_UC1))

myline = np.linspace(2, 9) # we use this to output evenly spaced numbers which helps us drawing a line 
plt.scatter(x, y, c='k') # again, these could have been named x_test, etc. but this is easier; the colour is black, think CMYK printing where black is the keyline k
plt.plot(myline, Equation_UC1(myline),label="linear Regression Results, of degree = 1")
# the following lines provide textual descriptions for the plot
plt.title('Training Data Use Case 1 - UC1')
plt.xlabel('hours of studying')
plt.ylabel('test results')
plt.legend()

# Plot the error lines
for i in range(len(x)):
    plt.plot([x[i], x[i]], [Equation_UC1(x[i]), y[i]], '--k')

# Show the plot
plt.show()
```

While this looks quite good, it would be convenient to quantify how good this model/approximation/*the fit* is.

We can't go too deep into the statistics foundation but will make use of some predefined functions.


Taking one step back, we want to answer students question about what exam result to expect. For our model, we only rely on one metric as an input which is the number of hours of studying.
We expect that we have this direct **corellation** between hours of studying and exam results. 
In other words, we consider the hours of studying an important input to get a good prediction. For this step, we have to **find a model** that maps any kind number a particular student gives us for the hours of studying to a realsitic output.

Alternatively, being lazy, we could also just provide the mean result as our prediction no matter what hour of studying a student tells us. 

Let's first calculate the mean of the test results that for our current data set (i.e. the training data set) to have a foundation to compare against. 

Another important metric is the *coefficient of determination* which is often abbreviated as *R-squared* 

$$R^2=\frac{\operatorname{Var}(\text { mean })-\operatorname{Var}(\text { line })}{\operatorname{Var}(\text { mean })}$$
$ \operatorname{Var}(\text { mean })$ is sum of the squared differences of the actual data values from the **mean**
$ \operatorname{Var}(\text { line })$ is sum of the squared differences of the actual data values from the **fitted line**
this is then normed through dividing by $ \operatorname{Var}(\text { mean })$

$R^2$ values are on a scale of zero to one or can be interpreted as a percentage.
For example, if $R^2 = 81\%$ this means that there is 81% less variation around the line than the mean. Or the given realtionship (hours of studying corellated with exam result) accounts for 81% of the variation. 
So most (81% to be exact) of the, variation in the data is explained by the hours/exam-results relationship. Which also means that our understanding that putting in more hours has a direct (but not 1:1) relationsihp with the exam-results is quite right.
We can also say the relationship between the two variables explains 81% of the variation in the data.
Again, we want to see if our identified realtionship is relevant. 
The disadvantage of $R^2$ is that squaring prevents us from saying the corellation is positive or negative. For many cases this is obvious. Not less studying means better exam results. 

```python
# Calculating the some metrics to gauge how well our model fits its underlying data 
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

print(f"The mean prediction (y-value) based on our current dataset (training) is:  {np.mean(x)}")

y_pred_UC1 = np.poly1d(Equation_UC1)
R2 = r2_score(y, y_pred_UC1(x))
print(f"The R^2 score for UC1 is {round(R2,2)} which is to be interpreted as a percentage of our relationship explaing the variation in data") 

rmse = sqrt(mean_squared_error(y, y_pred_UC1(x)))
print(f"The Root Mean Squared Error (RMSE) for UC1 is {round(rmse,2)} we can understand this as the non sign-cancelling avearge devidation from data points to the line, an absolute distance!") 
print()
```

<!-- #region -->
# Step 6: Adjust previous code degree of polynomial regression is easily adjustable

# Before we jump into UC2, let's condense previous findings (the plot with dotted lines and the previously generated metrics as one plot to compare against)



UC2=2
UC3=5

<!-- #endregion -->

```python
# Use Case 2 - Polynomial Regression
# Calculate the fit line with degree 2 which means a Quadratic Regression
Model_UC2 = np.polyfit(x, y, deg=2) # this is the regession and we specify the degree to be 1
Equation_UC2 = np.poly1d(Model_UC2) # poly1d helps us humans read the equation and makes several aspects callable from other functions

print(f"The equation for UC2 is y = {Equation_UC2}") # Still using f-notation

# myline = np.linspace(2, 9) # we use this to output evenly spaced numbers which helps us drawing a line 
plt.scatter(x, y, c='k') 
plt.plot(myline, Equation_UC2(myline),label="Quadratic Regression Results, degree = 2")
# the following lines provide textual descriptions for the plot
plt.title('Training Data Use Case 2 - UC2')
plt.xlabel('hours of studying')
plt.ylabel('test results')
plt.legend()

# Plot the error lines
for i in range(len(x)):
    plt.plot([x[i], x[i]], [Equation_UC2(x[i]), y[i]], '--k')

# Show the plot
plt.show()
```

```python
print(f"The mean prediction (y-value) based on our current dataset (training) has not changed an is still:  {np.mean(x)}")

y_pred_UC2 = np.poly1d(Equation_UC2)
R2 = r2_score(y, y_pred_UC2(x))
print(f"The R^2 score is {round(R2,2)} which is to be interpreted as a percentage of our relationship explaing the variation in data") 

rmse = sqrt(mean_squared_error(y, y_pred_UC2(x)))
print(f"The Root Mean Squared Error (RMSE) is {round(rmse,2)} we can understand this as the non sign-cancelling avearge devidation from data points to the line, an absolute distance!") 
print()
```

Note the curvature of the line.
Also note that the metrics haven't changed a lot, they even got worse (which is in part due to the small scale of our dataset)

```python
# Use Case 3 - Polynomial Regression with higher degree
# Calculate the fit line with degree 5
Model_UC3 = np.polyfit(x, y, deg=5) # this is the regession and we specify the degree to be 1
Equation_UC3 = np.poly1d(Model_UC3) # poly1d helps us humans read the equation and makes several aspects callable from other functions

print(f"The equation for UC3 is y = {Equation_UC3}") # Still using f-notation

# myline = np.linspace(2, 9) # we use this to output evenly spaced numbers which helps us drawing a line 
plt.scatter(x, y, c='k') 
plt.plot(myline, Equation_UC3(myline),label="Regression Results, degree = 5")
# the following lines provide textual descriptions for the plot
plt.title('Training Data Use Case 3 - UC3')
plt.xlabel('hours of studying')
plt.ylabel('test results')
plt.legend()

# Plot the error lines
for i in range(len(x)):
    plt.plot([x[i], x[i]], [Equation_UC3(x[i]), y[i]], '--k')

# Show the plot
plt.show()
```

```python
print(f"The mean prediction (y-value) based on our current dataset (training) has not changed an is still:  {np.mean(x)}")

y_pred_UC3 = np.poly1d(Equation_UC3)
R2 = r2_score(y, y_pred_UC3(x))
print(f"The R^2 score is {round(R2,2)} which is to be interpreted as a percentage of our relationship explaing the variation in data") 

rmse = sqrt(mean_squared_error(y, y_pred_UC3(x)))
print(f"The Root Mean Squared Error (RMSE) is {round(rmse,2)} we can understand this as the non sign-cancelling avearge devidation from data points to the line, an absolute distance!") 
print()
```

Note the hint that Python gives us: *RankWarning: Polyfit may be poorly conditioned*

Also note how this cuved line hits all the data points it *knew of*

This is reflected in our metrics where $R^2$ is 100% and the RMSE = 0.

Question: Did we generate the perfect model to predict exam outcomes based on historic data collection of hours spent for preparing the exam using the formula given by 'Equation_UC3'?


POP OUT BOX:
TRY IN THEIR OWN TIME DEGREE OF: 9, 17, 35


# Step 7: Taking one step back and comparing these results
We could dedut from our previous setps that the higher the degree, the better the prediction.

Let's see all three use-cases in one plot. 

And also get each model's prediction for a student enquiring about the predicted exam result based o **8h of studying**. This value wasn't part of our training dataset. 



```python
# Overview Plot

fig, ax = plt.subplots()

ax.scatter(x, y, c='k') 


ax.plot(myline, Equation_UC1(myline), color='blue', label="linear Regression Results, of degree = 1")
ax.plot(8,Equation_UC1(8),color='blue', marker='o', markersize=20)
ax.annotate(round(Equation_UC1(8),2),xy=(8,(Equation_UC1(8)-2)))
# Plot the error lines
for i in range(len(x)):
    ax.plot([x[i], x[i]], [Equation_UC1(x[i]), y[i]], '--k')


ax.plot(myline, Equation_UC2(myline),color='orange', label="linear Regression Results, of degree = 2")
ax.plot(8,Equation_UC2(8),color='orange', marker='o', markersize=20)
ax.annotate(round(Equation_UC2(8),2),xy=(8,(Equation_UC2(8)-2)))
# Plot the error lines
for i in range(len(x)):
    ax.plot([x[i], x[i]], [Equation_UC2(x[i]), y[i]], '--k')

ax.plot(myline, Equation_UC3(myline),color='green', label="linear Regression Results, of degree = 5")
ax.plot(8,Equation_UC3(8),color='green', marker='o', markersize=20)
ax.annotate(round(Equation_UC3(8),2),xy=(8,(Equation_UC3(8)-2)))
# Plot the error lines
for i in range(len(x)):
    ax.plot([x[i], x[i]], [Equation_UC3(x[i]), y[i]], '--k')

# Annotaiton as text box
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# plt.text(2, 15, f"points show predicted exam results based on 8h of studying per UC", fontsize=8, verticalalignment='top', bbox=props)



ax.axis(ymin=0,ymax=10)
ax.grid()
ax.set_xlabel('hours of studying')
ax.set_ylabel('test results')
ax.legend()
ax.legend()
ax.set_title('Overview, points show predicted exam results based on 8h of studying per UC')


```

# Step 8: Now put the testing dataset into place

```python

print(f"The x-values used to train our model on are: {x}") # Note the random sequence which is also good ML practice
print(f"The x-values used to test our model on are: {x_test}") # Note the random sequence which is also good ML practice
print(f"The predictions based on UC1 are: {y_pred_UC1(x_test)}") # Note the random sequence which is also good ML practice
print(f"The predictions based on UC2 are: {y_pred_UC2(x_test)}") # Note the random sequence which is also good ML practice
print(f"The predictions based on UC3 are: {y_pred_UC3(x_test)}") # Note the random sequence which is also good ML practice


# START UGLY COPY PASTE WORKAROUND

fig, ax = plt.subplots()

ax.scatter(x, y, c='k') 


ax.plot(myline, Equation_UC1(myline), color='blue', label="linear Regression Results, of degree = 1")
ax.plot(8,Equation_UC1(8),color='blue', marker='o', markersize=20)
ax.annotate(round(Equation_UC1(8),2),xy=(8,(Equation_UC1(8)-2)))
# Plot the error lines
for i in range(len(x)):
    ax.plot([x[i], x[i]], [Equation_UC1(x[i]), y[i]], '--k')


ax.plot(myline, Equation_UC2(myline),color='orange', label="linear Regression Results, of degree = 2")
ax.plot(8,Equation_UC2(8),color='orange', marker='o', markersize=20)
ax.annotate(round(Equation_UC2(8),2),xy=(8,(Equation_UC2(8)-2)))
# Plot the error lines
for i in range(len(x)):
    ax.plot([x[i], x[i]], [Equation_UC2(x[i]), y[i]], '--k')

ax.plot(myline, Equation_UC3(myline),color='green', label="linear Regression Results, of degree = 5")
ax.plot(8,Equation_UC3(8),color='green', marker='o', markersize=20)
ax.annotate(round(Equation_UC3(8),2),xy=(8,(Equation_UC3(8)-2)))
# Plot the error lines
for i in range(len(x)):
    ax.plot([x[i], x[i]], [Equation_UC3(x[i]), y[i]], '--k')

# Annotaiton as text box
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# plt.text(2, 15, f"points show predicted exam results based on 8h of studying per UC", fontsize=8, verticalalignment='top', bbox=props)



ax.axis(ymin=0,ymax=10)
ax.grid()
ax.set_xlabel('hours of studying')
ax.set_ylabel('test results')
ax.legend()
ax.legend()
ax.set_title('Overview, points show predicted exam results based on 8h of studying per UC')


# END UGLY COPY PASTE WORKAROUND





# ax.scatter(x, y, c='k') # again, these could have been named x_test, etc. but this is easier; the colour is black, think CMYK printing where black is the keyline k
ax.scatter(x_test, y_test, c='b')
# plt.show
# print (y_test)
# print(y_pred_UC1(x_test))
# print(y_pred_UC2(x_test))
# print(y_pred_UC3(x_test))
```

# Step 9: Compare training, testing: Over and underfitting
We could put all this into a nice Pandas dataframe
- Table Column 1 = Degree
- Table Column 2 = Training RMSE
- Table Column 3 = Training R^2
- Table Column 4 = Testing RMSE
- Table Column 5 = Testing  R^2



# Step 10: Reflect on these previous results
- while looking at training dataset in isolation, the higher the degree the better, as the curve hits all the points
- but as soon as we get to the testing dataset, we see discrepancy, that is called **overfitting**. 
- This means, our model that is overfitted to the training data set has a low **bias** (as it never systematically over or underestimates the values, especially true for models with high polynomial degree). From another perspective, we can say that they don't reflect the reality and are only overfitted to an artificial abstraction (for more complicated ML tasks it is also said that the ML model might *remember* the training set too well)
- Another important metric, the **variance** tells us how well the model performs across different datasets. Here, the linear regression is very good. We can also call it robust.
- So we have to consider all these aspects; and especially for more complex ML tasks where we can't just visualise the 2D or 3D relationship with a simple plot, we have to be very careful; in other words: We won't be able to tell as easily in the realm of ML what errors we made during the model creation for models invovling more dimensions (which most do)
- AND remember: *Garbage in, Garbage out* and *correlation isn't a sign for causality*, just because almost every winner in the olympic games drank water, drinking heaps of water won't make you an olympic winner.
  - for some ML tasks it is even worse: The telco provider who knows that some customers will cancel their contract in the next few months might not know at all: WHY and WHAT TO DO AGAINST THAT. 


# Step 11: Outlook
- for neuronal networks: non linear activation functions (ReLu, Sigmoid, ... ) 
- next step: classification and categorisation and clustering (the three c's)
