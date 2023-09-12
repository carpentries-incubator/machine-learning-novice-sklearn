---
title: "Regression"
teaching: 45
exercises: 15
questions:
- "What is supervised learning?"
- "How can I model data and make predictions using regression?"
objectives:
- "Apply linear regression with Scikit-Learn to create a model."
- "Measure the error between a regression model and real data."
- "Analyse and assess the accuracy of a linear model using Scikit-Learn's metrics library."
- "Understand how more complex models can be built with non-linear equations."
- "Apply polynomial modelling to non-linear data using Scikit-Learn."
keypoints:
- "Scikit-Learn is a Python library with lots of useful machine learning functions."
- "Scikit-Learn includes a linear regression function."
- "Scikit-Learn includes a polynomial modelling function which is useful for modelling non-linear data."
---

# Supervised learning

Classical machine learning is often divided into two categories – supervised and unsupervised learning. 

For the case of supervised learning we act as a "supervisor" or "teacher" for our ML algorithms by providing the algorithm with "labelled data" that contains example answers of what we wish the algorithm to achieve. 

For instance, if we wish to train our algorithm to distinguish between images of cats and dogs, we would provide our algorithm with images that have already been labelled as "cat" or "dog" so that it can learn from these examples. If we wished to train our algorithm to predict house prices over time we would provide our algorithm with example data of datetime values that are "labelled" with house prices.

Supervised learning is split up into two further categories: classification and regression. For classification the labelled data is discrete, such as the "cat" or "dog" example, whereas for regression the labelled data is continuous, such as the house price example.

In this episode we will explore how we can use regression to build a "model" that can be used to make predictions.


# Regression

Regression is a statistical technique that relates a dependent variable (a label in ML terms) to one or more independent variables. A regression model attempts to describe this relation by fitting the data as closely as possible according to mathematical criteria. This model can then be used to predict new labelled values by inputting the independent variables into it. For example, if we create a house price model we can then feed in any datetime value we wish, and get a new house price value prediction.

Regression can be as simple as drawing a "line of best fit" through data points, known as linear regression, or more complex models such as polynomial regression, and is used routinely around the world in both industry and research. You may have already used regression in the past without knowing that it is also considered a machine learning technique!

![Example of linear and polynomial regressions](../fig/regression_example.png)

## Linear regression using Scikit-Learn

We've had a lot of theory so time to start some actual coding! Let's create regression models for a very small dataset that will predict exam scores from hours spent revising. Exam scores is our labelled dependent variable, and hours spent revising is our independent input variable.

Let's define our dataset and visualise it:

~~~
import seaborn as sns

# Load in a bulk dataset from seaborn
# Anscomes Quartet consists of 4 sets of data
data = sns.load_dataset("anscombe")
print(data.head())

# Split out the 1st dataset from the total
data_1 = data[data["dataset"]=="I"]
data_1 = data_1.sort_values("x")

# Inspect the data
print(data_1.head())
~~~
{: .language-python}

~~~
import matplotlib.pyplot as plt

plt.scatter(data_1["x"], data_1["y"])
plt.xlabel("x")
plt.ylabel("y")
plt.show()
~~~
{: .language-python}

![Inspection of our dataset](../fig/regression_inspect.png)


Now lets import Scikit-Learn and use it to create a linear regression model. The Scikit-Learn `regression` function that we will use is designed for datasets where multiple parameters are used and so it expects to be given multi-dimensional array data. To get it to accept our single dimension data, we need to convert the simple lists to numpy arrays with numpy's `reshape` function.

~~~
import sklearn.linear_model as skl_lin
import numpy as np

x_data = np.array(x_data).reshape(-1, 1)
y_data = np.array(y_data).reshape(-1, 1)

lin_regress = skl_lin.LinearRegression().fit(x_data, y_data)
~~~
{: .language-python}

The mathematical equation for a linear fit is `y = mx + c` where `y` is our output exam result values, `x` is our input revision hour values, `m` represents the gradient of the linear fit, and `c` represents the intercept with the y-axis.

As well as using our newly created `lin_regress` model to predict new values, we can also inspect the fit coefficients using `.coef_` and `.intercept_`. The Scikit-Learn code is designed to calculate multiple coefficients and intercepts at once so these return values will be arrays. Since we've just got one parameter we can just grab the first item from each of these arrays as follows:


Once we have created the model using the `regression` function we can use Scikit-Learn's `predict` function to convert input values into predictions. 

~~~
import math
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def fit_a_linear_model(x, y):
    # Define our estimator/model
    model = LinearRegression(fit_intercept=True)

    # tweak our data to work with our estimator/model
    x_data = np.array(x).reshape(-1, 1)
    y_data = np.array(y).reshape(-1, 1)

    # train our estimator/model using our data
    lin_regress = model.fit(x_data,y_data)

    # inspect the trained estimator/model parameters
    m = lin_regress.coef_
    c = lin_regress.intercept_
    print("linear coefs=",m, c)

    # predict some values using our trained estimator/model
    # (in this case - our input data)
    linear_data = lin_regress.predict(x_data)

    # visualise!
    plt.plot(x_data, linear_data)
    plt.scatter(x_data, y_data)

    plt.xlabel("x")
    plt.ylabel("y")

    error = math.sqrt(mean_squared_error(y_data, linear_data))
    print("linear error=",error)

    return lin_regress
~~~
{: .language-python}

![Linear regression of our dataset](../fig/regress_linear.png)


This looks like a reasonably good fit to the data points, but rather than rely on our own judgement lets calculate the fit error instead. Scikit-Learn doesn't provide a root mean squared error function, but it does provide a mean squared error function. We can calculate the root mean squared error simply by taking the square root of the output of this function. The `mean_squared_error` function is part of the Scikit-Learn `metrics` module, so we'll have to add that to our imports as well as the `math` module:

Repeat for dataset2

~~~
# repeat for dataset 2
data_2 = data[data["dataset"]=="II"]
data_2 = data_2.sort_values("x")

fit_a_linear_model(data_2["x"],data_2["y"])

plt.show()
~~~
{: .language-python}

![Linear regression of our dataset](../fig/regress_linear_2nd.png)


## Polynomial regression using Scikit-Learn

Now that we have learnt how to do a linear regression it's time look into polynomial regressions. Polynomial functions are non-linear functions that are commonly-used to model data. Mathematically they have `N` degrees of freedom and they take the following form `y = a + bx + cx^2 + dx^3 ... + mx^N`

If we have a polynomial of degree N=1 we once again return to a linear equation `y = a + bx` or as it is more commonly written `y = mx + c`. Let's create a polynomial regression using N=2. In Scikit-Learn this is done in two steps. First we pre-process our input data `x_data` into a polynomial representation using the `PolynomialFeatures` function:

Then we can create our polynomial regressions using the `LinearRegression().fit()` function again, but this time using the polynomial representation of our `x_data` instead. As before we can also inspect the regression coefficients and the intercept gradient, noting that the polynomial expression has multiple coefficients.

~~~
from sklearn.preprocessing import PolynomialFeatures

def fit_a_poly_model(x,y):
    # Define our estimator/model(s)
    poly_features = PolynomialFeatures(degree=2)
    poly_regress = LinearRegression()

    # tweak our data to work with our estimator/model
    x_data = np.array(x).reshape(-1, 1)
    y_data = np.array(y).reshape(-1, 1)

    x_poly = poly_features.fit_transform(x_data)

    # define and train our model
    poly_regress.fit(x_poly,y_data)

    # inspect trained model parameters
    poly_m = poly_regress.coef_
    poly_c = poly_regress.intercept_
    print("poly_coefs",poly_m, poly_c)

    # predict some values using our trained estimator/model
    # (in this case - our input data)
    poly_data = poly_regress.predict(x_poly)

    # visualise!
    plt.plot(x_data, poly_data)

    poly_error = math.sqrt(mean_squared_error(y_data,poly_data))
    print("poly error=",poly_error)

    return poly_regress
~~~
{: .language-python}

We can once again use our model to convert input values into predictions. Lets plot our original data, linear model, and polynomial model together as well as compare the errors of the linear and polynomial fits.

~~~
fit_a_linear_model(data_2["x"],data_2["y"])
fit_a_poly_model(data_2["x"],data_2["y"])

plt.show()
~~~
{: .language-python}

![Comparison of the regressions of our dataset](../fig/regress_both.png)


Comparing the plots and errors it seems like a polynomial regression of N=2 fits the data better than a linear regression.

> ## Exercise: Try repeating your polynomial regression with different N values
> 1. What happens when you try using a polynomial of N=1?
> 2. What happens if you try using N=3 or more? Do the errors get better or worse?
{: .challenge}


> ## Exercise: How do models perform against new data?
> We now have some more exam score data that we can use to evaluate our existing models:
> ~~~
> x_new = [2.5, 4.5, 6.7, 8, 10, 11] # hours spent revising
> y_new = [5, 6, 8, 10, 11, 12]  # exam results
> ~~~
> {: .language-python}
>
> Try plotting this new data alongside our old data and existing regression models. Which model performs better at predicting these new values? What comments can you say about our original dataset? How could we improve our modelling attempts?
> > ## Solution
> > ![Existing plot with new dataset](../fig/regression_new_data.png)
> {: .solution}
{: .challenge}

When looking at our original dataset it seems the higher the degree of polynomial, the better the fit, as the curve hits all the points. But as soon as we input our new dataset we see that our models fail to predict the new results, and higher degree polynomials perform noticably worse than the original linear regression. This phenomenon is known as overfitting - our original models have become too specific to our original data and now lack the generality we expect from a model. You could say that our models have learnt the answers but failed to understand the assignment!


~~~
dataset = sns.load_dataset("penguins")
dataset.head()
~~~
{: .language-python}

~~~
dataset.dropna(inplace=True)
dataset.head()
~~~
{: .language-python}

~~~
dataset_1 = dataset[:146]

x_data = dataset_1["body_mass_g"]
y_data = dataset_1["bill_depth_mm"]
~~~
{: .language-python}

~~~
import matplotlib.pyplot as plt

trained_model = fit_a_linear_model(x_data, y_data)

plt.xlabel("mass g")
plt.ylabel("depth mm")
plt.show()
~~~
![Comparison of the regressions of our dataset](../fig/regress_penguin_lin.png)
~~~
x_data_all = dataset["body_mass_g"]
y_data_all = dataset["bill_depth_mm"]

x_data_all = np.array(x_data_all).reshape(-1,1)

y_predictions = trained_model.predict(x_data_all)

error = math.sqrt(mean_squared_error(y_data_all, y_predictions))
print("penguin error=",error)

plt.scatter(x_data_all, y_data_all)
plt.scatter(x_data, y_data)

plt.plot(x_data_all, y_predictions)
plt.xlabel("mass g")
plt.ylabel("depth mm")
plt.show()
~~~
![Comparison of the regressions of our dataset](../fig/regress_penguin_lin_tot.png)
{: .language-python}
Remember: *Garbage in, Garbage out* and *correlation does not equal causation*. Just because almost every winner in the olympic games drank water, it doesn't mean that drinking heaps of water will make you an olympic winner.

{% include links.md %}