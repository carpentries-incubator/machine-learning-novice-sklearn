::: {.cell .markdown}

------------------------------------------------------------------------

title: \"Regression and its use in Scikit Learn\" teaching: 45
exercises: 10 questions:

-   \"What is regression\"

-   \"\" objectives:

-   \"Recall that scikit-learn has built in linear regression
    functions.\"

-   \"Measure the error between a regression model and real data.\"

-   \"Apply scikit-learn\'s linear regression to create a model.\"

-   \"Analyse and assess the accuracy of a linear model using
    scikit-learn\'s metrics library.\"

-   \"Understand that more complex models can be built with non-linear
    equations.\"

-   \"Apply scikit-learn\'s polynomial modelling to non-linear data.\"
    keypoints:

-   \"Scikit Learn is a Python library with lots of useful machine
    learning functions.\"

-   \"Scikit Learn includes a linear regression function.\"

-   ## \"It also includes a polynomial modelling function which is useful for modelling non-linear data.\" {#it-also-includes-a-polynomial-modelling-function-which-is-useful-for-modelling-non-linear-data}
:::

::: {.cell .markdown}
# TO DO

-   Colab. Runtime. Change Runtime Type and select GPU or TPU
-   sequencing. Introduce the concept of line fitting first.
-   AND THINK ABOUT HOW TO STREAMLINE THE CODE, PUT MORE FUNCTIONS, SO
    CHANGES ARE SIMPLER (DEFINE A VARIABLE FOR THE DEGREE AND UE THE
    SAME ALL OVER AGAIN)
    -   this is especially good if we then ingest a massive datset
-   the residuals
-   then show the absolute numbers added up (we have a variance of 5.7
    for first plout with linear reg; then we have 5.2 for second one
    with degree 2; and ultimately we have a perfect fit for the 5 deg)
-   BUT then we do exactly the same stuff for the test sub-set and boom,
    what used to be the best is now totally gone. This is overfitting
-   ALSO introduce the average, or mean, to say the average outcome of
    the test was x; one could argue based on pure statistics that
    prodicing this predicition is better than prodicing no prediction at
    all, but this is completely independent from the hours; so someone
    asks how many what output to get for a then again for b, then for c
    hours and our answer is always the mean value. We can visualise this
    as a horizontal line where no matter what input x value, we always
    get the output value as the same, i.e. the mean
-   The inability for a machine learning method (like linear regression)
    to capture the true relationship is called bias., difference in fits
    between data sets is called Variance
-   see example of self-driving cars
    <https://github.com/stephencwelch/self_driving_cars>
:::

::: {.cell .markdown}
episode

-   single topic
-   with clear
    -   questions
    -   objectives
    -   key points The idea behind the name "episode" is the thought
        that each one should last about as long as an episode for an
        television series.
:::

::: {.cell .markdown}
-   trying to quantify how good a fit it is (some say: the accurarcy of
    a model)
    -   overfitting and underfitting
    -   R2 and other metrics
-   is extrapolation possible? AI - only replicating
    exsiting/philosophical discussion
:::

::: {.cell .markdown}
In general, Machine Learning is all about making **predictions** and
classifications. In this episode, we will focus on predictions using
regression, while **clasisfication** (supervised learning, based on
labeled data) and **clustering** (unsupervised learning, based on
unlabeled data) will be the topics of the next two upcoming episodes.

## Initial remarks

Here, we will start with a clean dataset. In real-life, data often needs
to be cleaned and wrangled (handling duplicates, incomplete datasets,
typos, \...). Usually, such preparation takes up a lot of time, in
comparison, the core ML-effort is often rather quickly done;
interpreting, testing, validating and fine-tuning the model is
frequently the tricky part.

**Limitation #1:** a model is an abstraction of reality and never
perfect (in many cases there is natural variance). The recap of your
favorite TV show\'s last season isn\'t the same as watching it in its
entirety. The model can be too simple or too complex ([bias-variance
tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff);
where bias = underfitting, variance=overfitting).

A good (i.e. low) **bias** tells us how well the model is able to
reflect the (training) datapoints. For the linear case, neither
systematically too low nor too high,. A good (i.e. low) **variance**
tells us how well a model fits *different* data sets. A common first
step is to determine how well the model approximates the test dataset.

**Limitation #2:** If ML can be useful for extrapolation, going beyond
the range of your historical data depends on the case.

While the mathematical details are interesting, we\'ll focus on the
application to our ML use cases.
:::

::: {.cell .markdown}
## What is regression

[Regression is used to predict a value (e.g., price) based on known
historical data](https://vas3k.com/blog/machine_learning/). Following a
more formal definition, regression tries to link a response (often
$y$)to an input (often $x$) variable. To us, the core idea is to
identify a trend in given data and use it. Alternatives are clustering
(see upcoming episode X).

Use-cases might be:

-   make predictions (classic ML: Based on many inputs our model
    ingests, get a predicted value for something we are looking for or
    *extrapolate*; given life expectancies in the past, how will it be
    in 10 years from now?),
-   calculate intermediate values (given some values, we get an
    algorithm where we put a number or interest in and get a result)
-   model something as a function (that we can continue to use (if a
    combustion engine\'s torque is known, get the horsepower from it).
-   Identify outliers (find credit card fraud, spam, etc.)
-   later on our ML journey, we will make use of Activation Functions,
    etc.

There are various forms of regression, some examples follow. Picking the
most appropriate one is a common challenge and requires close inspection
and testing. Applying these algorithms is deemed ML. These aspects are
true for many aspects of Machine Learning.
:::

::: {.cell .markdown}
## Scikit Learn

Scikit-Learn is a Python package designed to give access to well-known
machine learning algorithms within Python code, through a clean,
well-thought-out API. It has been built by hundreds of contributors from
around the world, and is used across industry and academia.

Scikit-Learn is built upon Python's [NumPy (Numerical
Python)](http://numpy.org/) and [SciPy (Scientific
Python)](https://scipy.org/) libraries, which enable efficient in-core
numerical and scientific computation within Python. As such,
scikit-learn is not specifically designed for extremely large datasets,
though there is some work in this area. For this introduction to ML we
are going to stick to processing of small to medium datasets with
Scikit-learn, without the need for a Graphical Processing Unit (GPU).

By using Scikit-Learn, we can build upon exsiting functions which makes
it easier for us as Python programmers to use machine learning
techniques without having to implement them from scratch.
:::

::: {.cell .markdown}
## Example

We will go through one relatively small example focusing especially on
the ML aspects. Scenario: In a universtiy setting, a good friend asks us
how much time to allocate to studying for a test we have already taken.
We asked several other friends who also passed the test about their
efforts. This results in two lists: The hours spent studying and the
results on a scale of 0-15 (where 15 is reflects the best result).

Let\'s start with our data and get a visual impression of the said data
firstly.

To increase the relation to upcoming Machine Learning tasks, we will
perform a split of our initial dataset into a training (where we train
the model on) and testing dataset (to evaluate the accuracy later on)
datasets. Here, we are going to pick 25% of all our values to be used
for testing. \"[An 80%, 20% split is
common](https://carpentries-incubator.github.io/machine-learning-novice-python/05-validation/index.html).\"
vs. [\"A split of \~70% training, 30% test is
common.\"](https://carpentries-incubator.github.io/machine-learning-novice-python/02-data/index.html)
:::

::: {.cell .code execution_count="14"}
``` python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# given data
x_data = [2,3,5,7,8,9]
y_data = [4,5,7,10,14,15]

# print(type(x_data))
plt.scatter(x_data, y_data, c='k')
plt.xlabel('hours of studying')
plt.ylabel('test results')
plt.show()

# we could write x_train, x_test, y_train, y_test but as we will mostly work with the training data for now, we just call x_train --> x and y_train --> y 
x, x_test, y, y_test = train_test_split(
x_data, y_data, test_size=0.25, random_state=42)
print(x)
# making this super neat: put an interactive selector for the data sub-set to be shown
```

::: {.output .display_data}
![](682bb58cfd2fa9c4d285c0c9c5a57da58fc91677.png)
:::

::: {.output .stream .stdout}
    [9, 5, 8, 7]
:::
:::

::: {.cell .markdown}
Upon close inspection, we might be tempted to see a line connecting all
these given dots. Let\'s use some libraries to find such a line. Note
that by identifying a line as a good approximation, we follow an
approach called linear regression and will refer to this as Use Case 1
(short: UC1) for variables names in our code.\
We could refer back to Maths and create low level code to generate
needed equations. We could also make use of Python functions which are
only dedicated to linear regression. But, we will interpret our linear
regression as a special case of polynomial regression\-\--which we will
refer to very soon. The form is $y = mx +b$
:::

::: {.cell .code execution_count="47"}
``` python
# Use Case 1 - Linear Regression 

# Calculate the fit line with degree 1 which means a LINEAR REGRESSION
fit_UC1 = np.polyfit(x, y, deg=1) # this is the regession
fit_fn_UC1 = np.poly1d(fit_UC1) # this is the equation or function that results from the equation
# coeff = fit_UC1[0]


print(f" The euqation for UC1 is y = {fit_fn_UC1}") # this outputs the equation we can use later on
myline = np.linspace(2, 9)
plt.scatter(x, y, c='k')
plt.title('Use Case 1 - UC1')
plt.xlabel('hours of studying')
plt.ylabel('test results')
# plt.plot(myline, fit_fn_UC1(myline),label="linear Regression Results, of degree = 1")
plt.legend()



# # Show the plot
plt.show()

# ##  Alternative using sklearn
from sklearn.metrics import r2_score

# predict = np.poly1d(coeff)
predict = np.poly1d(coeff)
print('this' + str(predict[0])) 


R2 = r2_score(y, predict(x))
print(R2) 

residual = 0
print(len(x))
for i in range(len(x)):
    residual += predict[i] - x[i]
    print(residual)




# print(I SHALL DO THIS TO EXPLAIN MORE)
# ou can use the keyword full=True when calling polyfit (see http://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html) to get the least-square error of your fit:

# coefs, residual, _, _, _ = np.polyfit(lengths, breadths, 1, full=True)
# You can get the same answer by doing:

# coefs = np.polyfit(lengths, breadths, 1)
# yfit = np.polyval(coefs,lengths)
# residual = np.sum((breadths-yfit)**2)
# or

# residual = np.std(breadths-yfit)**2 * len(breadths)
# Additionally, if you want to plot the residuals, you can do:

# coefs = np.polyfit(lengths, breadths, 1)
# yfit = np.polyval(coefs,lengths)
# plot(lengths, breadths-yfit)

```

::: {.output .stream .stderr}
    No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
:::

::: {.output .stream .stdout}
     The euqation for UC1 is y =  
    2.114 x - 3.829
:::

::: {.output .display_data}
![](49c1131076334aec61bcb1a027579280b74eef09.png)
:::

::: {.output .stream .stdout}
    this-3.8285714285714225
    0.954006968641115
    4
    -12.828571428571422
    -15.714285714285708
    -23.714285714285708
    -30.714285714285708
:::
:::

::: {.cell .markdown}
OK. While this looks good and is the result of a blackbox operation, we
can wonder about the following things

a\) Can we quantify how well does this line fits?

b\) Are there other (non linear) approximations that might work better

c\) How well does this perform on non-training (i.e., test data)

Let\'s firstly plot the deviation from our data points to the line and
sum these up. Wait. If we just sum these up, some negative and positive
deviations might cancel each other out. We should get rid of the signs
of the deviations. So we square them. But to get a better understanding
of how many untis these are *actually off*, we should get the
sqaure-root out of them and add them up. This is referred to as RMS or
root mean squared.
$$ \sqrt{\Sigma_{0}^{samlpes}(distance \ y-value \ to \ line)} $$
:::

::: {.cell .code execution_count="56"}
``` python
from sklearn.metrics import mean_squared_error
from math import sqrt
THESE HAVE TO BE ADJUSTED

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
outputMSE = mean_squared_error(y_true, y_pred)
print(f"this is the mean squared error {outputMSE}")
outputRMSE = sqrt(outputMSE)
print(f"this is the ROOT mean squared error {outputRMSE}")
```

::: {.output .stream .stdout}
    this is the mean squared error 0.375
    this is the ROOT mean squared error 0.6123724356957945
:::
:::

::: {.cell .code execution_count="87"}
``` python
# Calculate the fit line with degree 2 which means a POLYNOMIAL REGRESSION DEGREE 2
fit_UC2 = np.polyfit(x, y, deg=2) # this is the regession
fit_fn_UC2 = np.poly1d(fit_UC2) # this is the equation or function that results from the equation
print(f" y = {fit_fn_UC2}")

# fig, ax = plt.subplots()

plt.scatter(x, y, c='k')
plt.xlabel('hours of studying')
plt.ylabel('test results')
plt.plot(myline, fit_fn_UC2(myline),label="linear Regression Results, of degree = 2")
plt.plot(myline, fit_fn_UC1(myline),label="linear Regression Results, of degree = 1")
plt.grid()
plt.legend()

print(x)
print(fit_fn_UC1)

# Calculate the variance of each point
residuals_UC1 = y - np.polyval(fit_UC1, x)
# print('these are the residuals' +str(residuals))
sumres_UC1 = sum(abs(residuals_UC1))
# print('this is their sum' +str(sumres))
variances_UC1 = np.var(residuals_UC1)


# Calculate the variance of each point
residuals_UC2 = y - np.polyval(fit_UC2, x)
# print('these are the residuals' +str(residuals))
sumres_UC2 = sum(abs(residuals_UC2))
# print('this is their sum' +str(sumres))
variances_UC2 = np.var(residuals_UC2)


# Plot the data points, the variance lines, and the fit line
for i in range(len(x)):
    plt.plot([x[i], x[i]], [fit_fn_UC1(x[i]), y[i]], '--k')
# plt.plot(x, fit_fn_UC1(x), '-r')


for i in range(len(x)):
    plt.plot([x[i], x[i]], [fit_fn_UC2(x[i]), y[i]], '--g')



# # these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.55, f"Sum of residuals UC1 = {round(sumres_UC1, 2)}", transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
plt.text(0.05, 0.45, f"Sum of residuals UC2 = {round(sumres_UC2, 2)}", transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    
# plt.plot(x, fit_fn_UC2(x), '-r')
```

::: {.output .stream .stdout}
     y =          2
    0.09091 x + 0.8545 x + 0.3273
    [9, 5, 8, 7]
     
    2.114 x - 3.829
:::

::: {.output .execute_result execution_count="87"}
    Text(0.05, 0.45, 'Sum of residuals UC2 = 2.29')
:::

::: {.output .display_data}
![](0c7fc2f897329391bdf36cf49ed06fccd5270546.png)
:::
:::

::: {.cell .code execution_count="84"}
``` python

fit_UC3 = np.polyfit(x, y, deg=5) # this is the regession
fit_fn_UC3 = np.poly1d(fit_UC3) # this is the equation or function that results from the equation
print(f" y = {fit_fn_UC3}")
plt.scatter(x, y, c='k')
plt.xlabel('hours of studying')
plt.ylabel('test results')
plt.plot(myline, fit_fn_UC3(myline),label="linear Regression Results, of degree = 5")
plt.legend()
plt.title('Regression results with 5 degrees')
for i in range(len(x)):
    plt.plot([x[i], x[i]], [fit_fn_UC3(x[i]), y[i]], '--g')


# Calculate the variance of each point
residuals_UC3 = y - np.polyval(fit_UC3, x)
# print('these are the residuals' +str(residuals))
sumres_UC3 = sum(abs(residuals_UC3))
# print('this is their sum' +str(sumres))
variances_UC3 = np.var(residuals_UC3)

plt.text(0.05, 0.45, f"Sum of residuals UC3 = {round(sumres_UC3,2)}", transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)

# # Show the plot
plt.show()
```

::: {.output .stream .stderr}
    /Users/jbri364/opt/anaconda3/envs/mlcarp2/lib/python3.10/site-packages/IPython/core/interactiveshell.py:3460: RankWarning: Polyfit may be poorly conditioned
      exec(code_obj, self.user_global_ns, self.user_ns)
:::

::: {.output .stream .stdout}
     y =           5           4          3          2
    -0.00252 x + 0.01989 x + 0.1754 x - 0.3643 x - 13.67 x + 57.97
:::

::: {.output .display_data}
![](6d9835cd97000cd3a2b37b059254957f60f018fd.png)
:::
:::

::: {.cell .markdown}
MAKE THE SAME THING BUT NOW WITH THE TEST DATA
:::

::: {.cell .code}
``` python
# alternatively the whole thing with a coloured bar

import numpy as np
import matplotlib.pyplot as plt

# Create sample point cloud data set
X = np.linspace(-10, 10, 100)
Y = 3*X + 5 + np.random.normal(0, 3, 100)

# Fit a linear function to the data
coefficients = np.polyfit(X, Y, 1)
fitted_function = np.poly1d(coefficients)

# Calculate the variance of each point
Y_pred = fitted_function(X)
RSS = np.sum((Y - Y_pred)**2)
variances = (Y - Y_pred)**2 / RSS

# Plot the point cloud data with colors based on variance
plt.scatter(X, Y, c=variances, cmap='viridis')

# Plot the fitted function
plt.plot(X, fitted_function(X), 'k-', linewidth=2)

# Plot the variance of each point as dotted lines
for i in range(len(X)):
    plt.plot([X[i], X[i]], [Y[i], Y_pred[i]], 'b:', alpha=variances[i])

# Add colorbar and labels
# plt.rcParams['figure.figsize'] = [20, 15]
cbar = plt.colorbar()
cbar.set_label('Variance')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Point Cloud with Fitted Function and Variance')
plt.show()
plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
```

::: {.output .display_data}
![](38863111c91dca38a49882b03c4d5c382b86bbc8.png)
:::

::: {.output .execute_result execution_count="9"}
    <Figure size 1200x800 with 0 Axes>
:::

::: {.output .display_data}
    <Figure size 1200x800 with 0 Axes>
:::
:::

::: {.cell .markdown}
Since the slope is not 0, it means that knowing a mouse\'s weight will
help us make a guess about that mouse\'s size. SS(mean): not considering
any input-output relationship: How is the variance or the sumed up
distance to the avg. size (divided by the number of samples); to get an
avg value and not to total (on avg +-2cm not 200cm over 100 samples)

Because the fit of our line is better than the fit of just calculating
the average value (take a step back, we wanted to answer a colleague\'s
question of how to predict the size of a mouse. We consider its weight
an important input to get a good prediction; we could also just avg our
sample and give our that value without any other computation involved).
Because we have a better R2, we say that some of the variation in mouse
size is \"explained\" by taking mouse weight into account. R2 is the
variation around the mean, so how much variation we have in the first
place, minus the variation after we neatly fit our line, divided by the
var(mean); so if our line fits perfectly, we have zero var (fit); we get
an R2 of 1. If our line is very bad, we have more variation (fit); it
will be less than 1. R2 of 60% means that there is a 60% reduction in
variance if we take something else into account; or mouse weight
explains 60% of the var in mouse size
:::
