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


# Regression

In this episode, we will apply supervised learning via regression to predict how much time should be invested to prepare for an exam, and what result can be expected given the preparation time. To accomplish this we need to hypothesise the kind of input data we need, and then we need to obtain the data and build a useful model. We might hypothesise a correlation between the time spent studying for an exam and the result. For simplicity, we'll start with a clean, existing dataset, as collecting and consolidating data can be time-consuming. We'll then split this into training and testing datasets. We will specify basic hyperparamters but our model find the parameters during training, and then we can evaluate the accuracy/validity of our model. The good news is Scikit-Learn has many of the tools we need to do this.

# Our hypothesis
We might imagine there is more to getting a good grade than just the time spent studying (as not all study time is high-quality, and previous experience could play a big part in your success). We have a dataset containing variables relating to time spent studying and grades by former students. It's normally useful to visualise the data first to get an impression of trends or relationships which might provide useful context for training and evaluating our model.

# Visualising the data
Lets get some training data and visualise it:

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

# Splitting dataset into training and testing sets

```python
import numpy as np
from sklearn.model_selection import train_test_split # again, we use common ML libraries (yes, this task could be performed manually)

# we could write x_train, x_test, y_train, y_test but as we will mostly work with the training data for now, we just call x_train --> x and y_train --> y 
x, x_test, y, y_test = train_test_split( 
x_data, y_data, test_size=0.25, random_state=42)
print(f"The x-values used to train our model on are: {x}") # Note the random sequence which is also good ML practice
```

# Create a model
Based on the graph we created above, it looks like we can simplify the relationship with a straight line. This is referred to as linear regression.  Polynomial regression is a similar approach, but where a curved line is fitted. Linear regression can be consdiered a special case of polynomial regression requiring fewer *terms*, which influence the shape of the line.

To create a linear regression model in Python:

```python
# Linear Regression, or 'Use Case 1' (UC1)
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

This looks quite good but it would be better to quantify how well this model fits the data. For our model, we're relying on one input (the number of hours of studying). In other words, we consider the hours of studying to be an important input to get a good prediction. 

At this point, it would be useful to consider the coefficient of determination (often abbreviated as R-squared): 

$$R^2=\frac{\operatorname{Var}(\text { mean })-\operatorname{Var}(\text { line })}{\operatorname{Var}(\text { mean })}$$
$ \operatorname{Var}(\text { mean })$ is sum of the squared differences of the actual data values from the **mean**
$ \operatorname{Var}(\text { line })$ is sum of the squared differences of the actual data values from the **fitted line**
this is then normed through dividing by $ \operatorname{Var}(\text { mean })$

$R^2$ values are on a scale of zero to one or can be interpreted as a percentage.
For example, if $R^2 = 81\%$ this means that there is 81% less variation around the line than the mean. Or the given realtionship (hours of studying corellated with exam result) accounts for 81% of the variation. 
This means putting in more hours has a direct (but not 1:1) relationsihp with the exam-results.
We can also say the relationship between the two variables explains 81% of the variation in the data.

We need to calculate the mean of the test results for the training data set to have a foundation to compare against.

```python
# Calculating some metrics to gauge how well our model fits its underlying data 
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

The curved line hits all the data points and this is reflected in our metrics where $R^2$ is 100% and the RMSE = 0.

Question: Did we generate the perfect model to predict exam outcomes based on historic data collection of hours spent for preparing the exam using the formula given by 'Equation_UC3'?


POP OUT BOX:
TRY IN THEIR OWN TIME DEGREE OF: 9, 17, 35


# Comparing our results
Let's see all three use-cases in one plot to compare them. We can also get each model's prediction for a student enquiring about the predicted exam result based on 8h of studying. This value wasn't part of our training dataset. 

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

# Using the testing dataset

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

# Compare training and testing results
We could put all this into a nice Pandas dataframe
- Table Column 1 = Degree
- Table Column 2 = Training RMSE
- Table Column 3 = Training R^2
- Table Column 4 = Testing RMSE
- Table Column 5 = Testing  R^2



# Reflections
When looking at the training dataset in isolation, it seems the higher the degree the better, as the curve hits all the points. But as soon as we get to the testing dataset, we see a discrepancy (called overfitting). This means our overfitted model has a low bias as it never systematically over or underestimates the values. From another perspective, we can say that they don't reflect reality and are only overfitted to an artificial abstraction. 

Another important metric, the variance, tells us how well the model performs across different datasets. Here the linear regression is very good. So we should consider all of these aspects when evaluating our results. For more complex ML tasks where we can't just visualise the 2D or 3D relationship with a simple plot, we have to be very careful. We won't be able to tell as easily what errors we made during the model creation for models invovling more dimensions.

Remember: *Garbage in, Garbage out* and *correlation does not equal causation*. Just because almost every winner in the olympic games drank water, it doesn't mean that drinking heaps of water will make you an olympic winner.

# Outlook
- for neuronal networks: non linear activation functions (ReLu, Sigmoid, ... ) 
- next step: classification and categorisation and clustering (the three c's)
