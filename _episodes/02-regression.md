---
title: "Regression"
teaching: 45
exercises: 30
questions:
- "How can I make linear regression models from data?"
- "How can I use logarithmic regression to work with non-linear data?"
objectives:
- "Learn how to use linear regression to produce a model from data."
- "Learn how to model non-linear data using a logarithmic."
- "Learn how to measure the error between the original data and a linear model."
keypoints:
- "We can model linear data using a linear or least squares regression."
- "A linear regression model can be used to predict future values."
- "We should split up our training dataset and use part of it to test the model."
- "For non-linear data we can use logarithms to make the data linear."
---

# Linear regression

If we take two variable and graph them against each other we can look for relationships between them. Once this relationship is established we can use that to produce a model which will help us predict future values of one variable given the other.

If the two variables form a linear relationship (a straight line can be drawn to link them) then we can create a linear equation to link them. This will be of the form y = m * x + c, where x is the variable we know, y is the variable we're calculating, m is the slope of the line linking them and c is the point at which the line crosses the y axis (where x = 0).

Using the Gapminder website we can graph all sorts of data about the development of different countries. Lets have a look at the change in [life expectancy over time in the United Kingdom](https://www.gapminder.org/tools/#$state$time$value=2018&showForecast:true&delay:100;&entities$filter$;&dim=geo;&marker$select@$geo=gbr&trailStartTime=1800;;&axis_x$which=time&domainMin:null&domainMax:null&zoomedMin=1800&zoomedMax=2018&scaleType=time&spaceRef:null;&axis_y$domainMin:null&domainMax:null&zoomedMin:1&zoomedMax:84.17&spaceRef:null;&size$domainMin:null&domainMax:null&extent@:0.022083333333333333&:0.4083333333333333;;&color$which=world_6region;;;&chart-type=bubbles).

Since around 1950 life expectancy appears to be increasing with a pretty straight line in other words a linear relationship. We can use this data to try and calculate a line of best fit that will attempt to draw a perfectly straight line through this data. One method we can use is called [linear regression or least square regression](https://www.mathsisfun.com/data/least-squares-regression.html). The linear regression will create a linear equation that minimises the average distance from the line of best fit to each point in the graph. It will calculate the values of m and c for a linear equation for us. We could do this manually, but lets use Python to do it for us.


## Coding a linear regression with Python
We'll start by creating a toy dataset of x and y coordinates that we can model.
~~~
x_data = [2,3,5,7,9]
y_data = [4,5,7,10,15]
~~~
{: .language-python}

We can use the `least_squares()` helper function to calculate a line of best fit through this data. 

Let's take a look at the math required to fit a line of best fit to this data. Open `regression_helper_functions.py` and view the code for the `least_squares()` function. 
~~~
def least_squares(data: List[List[float]]) -> Tuple[float, float]:
    """
    Calculate the line of best fit for a data matrix of [x_values, y_values] using 
    ordinary least squares optimization.

    Args:
        data (List[List[float]]): A list containing two equal-length lists, where the 
        first list represents x-values and the second list represents y-values.

    Returns:
        Tuple[float, float]: A tuple containing the slope (m) and the y-intercept (c) of 
        the line of best fit.
    """
    x_sum = 0
    y_sum = 0
    x_sq_sum = 0
    xy_sum = 0

    # Ensure the list of data has two equal-length lists
    assert len(data) == 2
    assert len(data[0]) == len(data[1])

    n = len(data[0])
    # Least squares regression calculation
    for i in range(0, n):
        if isinstance(data[0][i], str):
            x = int(data[0][i])  # Convert date string to int
        else:
            x = data[0][i]  # For GDP vs. life-expect data
        y = data[1][i]
        x_sum = x_sum + x
        y_sum = y_sum + y
        x_sq_sum = x_sq_sum + (x ** 2)
        xy_sum = xy_sum + (x * y)

    m = ((n * xy_sum) - (x_sum * y_sum))
    m = m / ((n * x_sq_sum) - (x_sum ** 2))
    c = (y_sum - m * x_sum) / n

    print("Results of linear regression:")
    print("m =", format(m, '.5f'), "c =", format(c, '.5f'))

    return m, c
~~~
{: .language-python}

The equations you see in this function are derived using some calculus. Specifically, to find a slope and y-intercept that minimizes the sum of squared errors (SSE), we have to take the partial derivative of SSE w.r.t. both of the model's parameters — slope and y-intercept. We can set those partial derivatives to zero (where the rate of SSE change goes to zero) to find the optimal values of these model coefficients (a.k.a parameters a.k.a. weights). 

To see how ordinary least squares optimization is fully derived, visit: [https://are.berkeley.edu/courses/EEP118/current/derive_ols.pdf](https://are.berkeley.edu/courses/EEP118/current/derive_ols.pdf)
~~~
from regression_helper_functions import least_squares
m, b = least_squares([x_data,y_data])
~~~
{: .language-python}

~~~
Results of linear regression:
m = 1.51829 c = 0.30488
~~~
{: .output}

We can use our new model to generate a line that predicts y-values at all x-coordinates fed into the model. Open `regression_helper_functions.py` and view the code for the `get_model_predictions()` function. Find the FIXME tag in the function, and fill in the missing code to output linear model predicitons.
~~~
def get_model_predictions(x_data: List[float], m: float, c: float) -> List[float]:
    """
    Calculate linear model predictions (y-values) for a given list of x-coordinates using 
    the provided slope and y-intercept.

    Args:
        x_data (List[float]): A list of x-coordinates for which predictions are calculated.
        m (float): The slope of the linear model.
        c (float): The y-intercept of the linear model.

    Returns:
        List[float]: A list of predicted y-values corresponding to the input x-coordinates.
    """
    linear_preds = []
    for x in x_data:
        # FIXME: Uncomment below line and complete the line of code to get a model prediction from each x value
#         y = _______
        # ANSWER
        y = m * x + c
        
        #add the result to the linear_data list
        linear_preds.append(y)
    return(linear_preds)
~~~
{: .language-python}

Using this function, let's return the model's predictions for the data we used to fit the model (i.e., the line of best fit). The data used to fit or train a model is referred to as the model's training dataset.

~~~
from regression_helper_functions import get_model_predictions
y_preds = get_model_predictions(x_data, m, b)
~~~
{: .language-python}

We can now plot our model predictions along with the actual data using the `make_regression_graph()` function.

~~~
from regression_helper_functions import make_regression_graph
help(make_regression_graph)
~~~
{: .language-python}

~~~
Help on function make_regression_graph in module regression_helper_functions:

make_regression_graph(x_data: List[float], y_data: List[float], y_pred: List[float], axis_labels: Tuple[str, str]) -> None
    Plot data points and a model's predictions (line) on a graph.
    
    Args:
        x_data (List[float]): A list of x-coordinates for data points.
        y_data (List[float]): A list of corresponding y-coordinates for data points.
        y_pred (List[float]): A list of predicted y-values from a model (line).
        axis_labels (Tuple[str, str]): A tuple containing the labels for the x and y axes.
    
    Returns:
        None: The function displays the plot but does not return a value.
~~~
{: .output}

~~~
make_regression_graph(x_data, y_data, y_preds, ['X', 'Y'])
~~~
{: .language-python}

### Testing the accuracy of a linear regression model
We now have a linear model for some training data. It would be useful to assess how accurate that model is. 

One popular measure of a model's error is the Root Mean Squared Error (RMSE). RMSE is expressed in the same units as the data being measured. This makes it easy to interpret because you can directly relate it to the scale of the problem. For example, if you're predicting house prices in dollars, the RMSE will also be in dollars, allowing you to understand the average prediction error in a real-world context.

To calculate the RMSE, we:
1. Calculate the sum of squared differences (SSE) between observed values of y and predicted values of y: `SSE = (y-y_pred)**2`
2. Convert the SSE into the mean-squared error by dividing by the total number of obervations, n, in our data: `MSE = SSE/n`
3. Take the square root of the MSE: `RMSE = math.sqrt(MSE)`
   
The RMSE gives us an overall error number which we can then use to measure our model’s accuracy with. 

Open `regression_helper_functions.py` and view the code for the `measure_error()` function. Find the FIXME tag in the function, and fill in the missing code to calculate RMSE.

~~~
import math
def measure_error(y: List[float], y_pred: List[float]) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) of a model's predictions.

    Args:
        y (List[float]): A list of actual (observed) y values.
        y_pred (List[float]): A list of predicted y values from a model.

    Returns:
        float: The RMSE (root mean square error) of the model's predictions.
    """
    assert len(y)==len(y_pred)
    err_total = 0
    for i in range(0,len(y)):
        # add up the squared error for each observation
        # FIXME: Uncomment the below line and fill in the blank to add up the squared error for each observation
#         err_total = err_total + ________
        # SOLUTION
        err_total = err_total + (y[i] - y_pred[i])**2

    err = math.sqrt(err_total / len(y))
    return err
~~~
{: .language-python}

~~~
import math
def measure_error(data1, data2):
    """Calculating RMSE (root mean square error) of model."""
    
    assert len(data1) == len(data2)
    err_total = 0
    for i in range(0, len(data1)):
        # FIXME: Uncomment the below line and fill in the blank to add up the squared error for each observation
#         err_total = err_total + ________
        err_total = err_total + (data1[i] - data2[i]) ** 2

    err = math.sqrt(err_total / len(data1))
    return err
~~~
{: .language-python}

Using this function, let's calculate the error of our model in term's of its RMSE. Since we are calculating RMSE on the same data that was used to fit or "train" the model, we call this error the model's training error.
~~~
from regression_helper_functions import measure_error
print(measure_error(y_data,y_preds))
~~~
{: .language-python}
~~~
0.7986268703523449
~~~
{: .output}

This will output an error of 0.7986268703523449, which means that on average the difference between our model and the real values is 0.7986268703523449. The less linear the data is the bigger this number will be. If the model perfectly matches the data then the value will be zero.

> ## Model Parameters (a.k.a. coefs or weights) VS Hyperparameters
> Model parameters/coefficients/weights are parameters that are learned during the model-fitting stage. That is, they are estimated from the data. How many parameters does our linear model have? In addition, what hyperparameters does this model have, if any?
> 
> > ## Solution
> > In a univariate linear model (with only one variable predicting y), the two parameters learned from the data include the model's slope and its intercept. One hyperparameter of a linear model is the number of variables being used to predict y. In our previous example, we used only one variable, x, to predict y. However, it is possible to use additional predictor variables in a linear model (e.g., multivariate linear regression).
> {: .solution}
{: .challenge}

### Predicting life expectancy

Now lets try and model some real data with linear regression. We'll use the [Gapminder Foundation's](http://www.gapminder.org) life expectancy data for this. Click [here](../data/gapminder-life-expectancy.csv) to download it, and place the file in your project's data folder (e.g., `data/gapminder-life-expectancy.csv`)

Let's start by reading in the data and examining it.
~~~
import pandas as pd
import numpy as np
df = pd.read_csv("data/gapminder-life-expectancy.csv", index_col="Life expectancy")
df.head()
~~~
{: .language-python}

Looks like we have data from 1800 to 2016. Let's check how many countries there are.
~~~
print(df.index) # There are 243 countries in this dataset. 
~~~
{: .language-python}

Let's try to model life expectancy as a function of time for individual countries. To do this, review the 'process_life_expectancy_data()' function found in regression_helper_functions.py. Review the FIXME tags found in the function and try to fix them. Afterwards, use this function to model life expectancy in the UK between the years 1950 and 1980. How much does the model predict life expectancy to increase or decrease per year?
~~~
def process_life_expectancy_data(filename, country, train_data_range, test_data_range=None):
    """Model and plot life expectancy over time for a specific country. Model is fit to data spanning train_data_range, and tested on data spanning test_data_range"""

    # Extract date range used for fitting the model
    min_date_train = train_data_range[0]
    max_date_train = train_data_range[1]
    
    # Read life expectancy data
    df = pd.read_csv(filename, index_col="Life expectancy")

    # get the data used to estimate line of best fit (life expectancy for specific country across some date range)
    # we have to convert the dates to strings as pandas treats them that way
    y_data_train = df.loc[country, str(min_date_train):str(max_date_train)]

    # create a list with the numerical range of min_date to max_date
    # we could use the index of life_expectancy but it will be a string
    # we need numerical data
    x_data_train = list(range(min_date_train, max_date_train + 1))

    # calculate line of best fit
    # FIXME: Uncomment the below line of code and fill in the blank
#     m, c = _______([x_data_train, y_data_train])
    m, c = least_squares([x_data_train, y_data_train])

    # Get model predictions for test data. 
    # FIXME: Uncomment the below line of code and fill in the blank 
#     y_preds_train = _______(x_data_train, m, c)
    y_preds_train = get_model_predictions(x_data_train, m, c)
    
    # FIXME: Uncomment the below line of code and fill in the blank
#     train_error = _______(y_data_train, y_preds_train)
    train_error = measure_error(y_data_train, y_preds_train)    
    print("Train RMSE =", format(train_error,'.5f'))
    make_regression_graph(x_data_train, y_data_train, y_preds_train, ['Year', 'Life Expectancy'])
    
    # Test RMSE
    if test_data_range is not None:
        min_date_test = test_data_range[0]
        if len(test_data_range)==1:
            max_date_test=min_date_test
        else:
            max_date_test = test_data_range[1]
        x_data_test = list(range(min_date_test, max_date_test + 1))
        y_data_test = df.loc[country, str(min_date_test):str(max_date_test)]
        y_preds_test = get_model_predictions(x_data_test, m, c)
        test_error = measure_error(y_data_test, y_preds_test)    
        print("Test RMSE =", format(test_error,'.5f'))
        make_regression_graph(x_data_train+x_data_test, pd.concat([y_data_train,y_data_test]), y_preds_train+y_preds_test, ['Year', 'Life Expectancy'])

    return m, c
~~~
{: .language-python}

Let's use this function to model life expectancy in the UK between the years 1950 and 1980. How much does the model predict life expectancy to increase or decrease per year?

~~~
from regression_helper_functions import process_life_expectancy_data
m, c = process_life_expectancy_data("data/gapminder-life-expectancy.csv",
                             "United Kingdom", [1950, 1980])
~~~
{: .language-python}

~~~
Results of linear regression:
m = 0.13687 c = -197.61772
Train RMSE = 0.32578
~~~
{: .output}

Let's see how the model performs in terms of its ability to predict future years. Run the `process_life_expectancy_data()` function again using the period 1950-1980 to train the model, and the period 2010-2016 to test the model's performance on unseen data.

~~~
m, c = process_life_expectancy_data("data/gapminder-life-expectancy.csv",
                             "United Kingdom", [1950, 1980], test_data_range=[2010,2016])
~~~
{: .language-python}

When we train our model using data between 1950 and 1980, we aren't able to accurately predict life expectancy in later decades. To explore this issue further, try out the excercise in the following section

> ## Models Fit Their Training Data — For Better Or Worse
> What happens to the test RMSE as you extend the training data set to include additional dates? Try out a couple of ranges  (e.g., 1950:1990, 1950:2000, 1950:2005); Explain your observations.
> 
> > ## Solution
> > ~~~
> > end_date_ranges = [1990, 2000, 2005]
> > for end_date in end_date_ranges:
> >     print('Training Data = 1950:' + str(end_date))
> >     m, c = process_life_expectancy_data("data/gapminder-life-expectancy.csv",
> >                                  "United Kingdom", [1950, end_date], test_data_range=[2010,2016])
> > ~~~
> > {: .language-python}
> > 
> > - Models aren't magic. They will take the shape of their training data. 
> > - When modeling time-series trends, it can be difficult to see longer-duration cycles in the data when we look at only a small window of time.
> > - If future data doesn't follow the same trends as the past, our model won't perform well on future data.
> {: .solution}
{: .challenge}

> ## Predicting Life Expectancy
> 1) Model Germany's predicted life expectancy between the years 1950 and 2000. What is the value of and c?
> 
> 2) Use the linear model you’ve just created to predict life expectancy in Germany for every year between 2001 and 2016. How accurate are your answers? If you worked for a pension scheme would you trust your answers to predict the future costs for paying pensioners?
> > ## Solution
> > ~~~
> > m,c = process_life_expectancy_data("data/gapminder-life-expectancy.csv", "Germany", [1950, 2000])
> > 
> > for x in range(2001,2017):
> >     print(x,0.212219909502 * x - 346.784909502)
> > ~~~
> > {: .language-python}
> > 
> > ~~~
> > df = pd.read_csv('data/gapminder-life-expectancy.csv',index_col="Life expectancy")
> > for x in range(2001,2017):
> >     y = m*x + c
> >     real = df.loc['Germany', str(x)]
> >     print(x, "Predicted", y, "Real", real, "Difference", y-real)
> >     
> > ~~~
> > {: .language-python}
> >
> > Predicted answers
> > ~~~
> > 2001 Predicted 77.86712941175517 Real 78.4 Difference -0.5328705882448332
> > 2002 Predicted 78.07934932125704 Real 78.6 Difference -0.5206506787429532
> > 2003 Predicted 78.29156923075897 Real 78.8 Difference -0.5084307692410306
> > 2004 Predicted 78.50378914026084 Real 79.2 Difference -0.6962108597391676
> > 2005 Predicted 78.71600904976276 Real 79.4 Difference -0.683990950237245
> > 2006 Predicted 78.92822895926463 Real 79.7 Difference -0.7717710407353735
> > 2007 Predicted 79.1404488687665 Real 79.9 Difference -0.7595511312335077
> > 2008 Predicted 79.35266877826842 Real 80.0 Difference -0.6473312217315765
> > 2009 Predicted 79.56488868777029 Real 80.1 Difference -0.5351113122297022
> > 2010 Predicted 79.77710859727222 Real 80.3 Difference -0.5228914027277796
> > 2011 Predicted 79.98932850677409 Real 80.5 Difference -0.5106714932259138
> > 2012 Predicted 80.20154841627601 Real 80.6 Difference -0.3984515837239826
> > 2013 Predicted 80.41376832577788 Real 80.7 Difference -0.2862316742221225
> > 2014 Predicted 80.6259882352798 Real 80.7 Difference -0.07401176472019699
> > 2015 Predicted 80.83820814478167 Real 80.8 Difference 0.03820814478167733
> > 2016 Predicted 81.0504280542836 Real 80.9 Difference 0.1504280542835943
> > ~~~
> > {: .output}
> >
> > Answers are between 0.15 years over and 0.77 years under the reality.
> > If this was being used in a pension scheme it might lead to a slight under prediction of life expectancy and cost the pension scheme a little more than expected.
> {: .solution}
{: .challenge}

# Logarithmic Regression

We've now seen how we can use linear regression to make a simple model and use that to predict values, but what do we do when the relationship between the data isn't linear?

As an example lets take the relationship between income (GDP per Capita) and life expectancy. The gapminder website will [graph](https://www.gapminder.org/tools/#$state$time$value=2017&showForecast:true&delay:206.4516129032258;&entities$filter$;&dim=geo;&marker$axis_x$which=life_expectancy_years&domainMin:null&domainMax:null&zoomedMin:45&zoomedMax:84.17&scaleType=linear&spaceRef:null;&axis_y$which=gdppercapita_us_inflation_adjusted&domainMin:null&domainMax:null&zoomedMin:115.79&zoomedMax:144246.37&spaceRef:null;&size$domainMin:null&domainMax:null&extent@:0.022083333333333333&:0.4083333333333333;;&color$which=world_6region;;;&chart-type=bubbles) this for us.

> ## Logarithms Introduction
> Logarithms are the inverse of an exponent (raising a number by a power).
> ```
> log b(a) = c
> b^c = a
> ```
> For example:
> ```
> 2^5 = 32
> log 2(32) = 5
> ```
> If you need more help on logarithms see the [Khan Academy's page](https://www.khanacademy.org/math/algebra2/exponential-and-logarithmic-functions/introduction-to-logarithms/a/intro-to-logarithms)
{: .callout}


The relationship between these two variables clearly isn't linear. But there is a trick we can do to make the data appear to be linear, we can take the logarithm of the Y axis (the GDP) by clicking on the arrow on the left next to GDP/capita and choosing log. [This graph](https://www.gapminder.org/tools/#$state$time$value=2017&showForecast:true&delay:206.4516129032258;&entities$filter$;&dim=geo;&marker$axis_x$which=life_expectancy_years&domainMin:null&domainMax:null&zoomedMin:45&zoomedMax:84.17&scaleType=linear&spaceRef:null;&axis_y$which=gdppercapita_us_inflation_adjusted&domainMin:null&domainMax:null&zoomedMin:115.79&zoomedMax:144246.37&scaleType=log&spaceRef:null;&size$domainMin:null&domainMax:null&extent@:0.022083333333333333&:0.4083333333333333;;&color$which=world_6region;;;&chart-type=bubbles) now appears to be linear.


## Coding a logarithmic regression

### Downloading the data

Download the GDP data from [http://scw-aberystwyth.github.io/machine-learning-novice/data/worldbank-gdp.csv](http://scw-aberystwyth.github.io/machine-learning-novice/data/worldbank-gdp.csv)

### Loading the data
Let's start by reading in the data. We'll collect GDP and life expectancy from two separate files using the read_data() function stored in regression_helper_functions.py. Use the read_data function to get GDP and life-expectancy in the year 1980 from all countries that have this data available.

~~~
from regression_helper_functions import read_data
data = read_data("data/worldbank-gdp.csv",
             "data/gapminder-life-expectancy.csv", "1980")
data
~~~
{: .language-python}

### Model GDP vs Life Expectancy
Review the `process_lifeExpt_gdp_data()` function found in `regression_helper_functions.py`. Review the FIXME tags found in the function and try to fix them. Afterwards, use this function to model life-expectancy versus GDP for the year 1980.
~~~
def process_lifeExpt_gdp_data(gdp_file, life_expectancy_file, year):
    """Model and plot life expectancy vs GDP in a specific year."""
    data = read_data(gdp_file, life_expectancy_file, year)

    gdp = data["GDP"].tolist()
    gdp_log = data["GDP"].apply(math.log).tolist()
    life_exp = data["Life Expectancy"].tolist()

    m, c = least_squares([life_exp, gdp_log])

    # model predictions on transformed data
    gdp_preds = []
    # list for plotting model predictions on top of untransformed GDP. For this, we will need to transform the model's predicitons.
    gdp_preds_transformed = []
    for x in life_exp:
        y_pred = m * x + c
        gdp_preds.append(y_pred)
        # FIXME: Uncomment the below line of code and fill in the blank
#         y_pred = math._______
        y_pred = math.exp(y_pred)
        gdp_preds_transformed.append(y_pred)

    # plot both the transformed and untransformed data
    make_regression_graph(life_exp, gdp_log, gdp_preds, ['Life Expectancy', 'log(GDP)'])
    make_regression_graph(life_exp, gdp, gdp_preds_transformed, ['Life Expectancy', 'GDP'])

    train_error = measure_error(gdp_preds, gdp)
    print("Train RMSE =", format(train_error,'.5f'))
~~~
{: .language-python}

Let's use this function to model life expectancy versus GDP for the year 1980. How accurate is our model in predicting GDP from life expectancy? How much is GDP predicted to grow as life expectancy increases? 

~~~
from regression_helper_functions import process_lifeExpt_gdp_data
process_lifeExpt_gdp_data("data/worldbank-gdp.csv",
             "data/gapminder-life-expectancy.csv", "1980")
~~~
{: .language-python}

On average, our model over or underestimates GDP by 14233.73. GDP is predicted to grow by .128 for each year added to life.

> ## Removing outliers from the data
> The correlation of GDP and life expectancy has a few big outliers that are probably increasing the error rate on this model. These are typically countries with very high GDP and sometimes not very high life expectancy. These tend to be either small countries with artificially high GDPs such as Monaco and Luxemborg or oil rich countries such as Qatar or Brunei. Kuwait, Qatar and Brunei have already been removed from this data set, but are available in the file worldbank-gdp-outliers.csv. Try experimenting with adding and removing some of these high income countries to see what effect it has on your model's error rate.
> Do you think its a good idea to remove these outliers from your model?
> How might you do this automatically?
{: .challenge}

{% include links.md %}
