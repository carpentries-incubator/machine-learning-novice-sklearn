---
title: "Instructor Notes"
---

# Linear regression

* helps us link two variables
* creates line of best fit
* show gapminder example of life epxectancy in UK
* straight line since 1950
* mathsisfun link for regression 

linear regression code

```def least_squares(data):```
```    x_sum = 0```
```    y_sum = 0```
```    x_sq_sum = 0```
```    xy_sum = 0```

```    assert len(data[0]) == len(data[1])```
```    assert len(data) == 2```

```    n = len(data[0])```
```    for i in range(0, n):```
```        x = int(data[0][i])```
```        y = data[1][i]```
```        x_sum = x_sum + x```
```        y_sum = y_sum + y```
```        x_sq_sum = x_sq_sum + (x**2)```
```        xy_sum = xy_sum + (x*y)```

```    m = ((n * xy_sum) - (x_sum * y_sum))```
```    m = m / ((n * x_sq_sum) - (x_sum ** 2))```
```    c = (y_sum - m * x_sum) / n```

```    print("Results of linear regression:")```
```    print("x_sum=", x_sum, "y_sum=", y_sum, "x_sq_sum=", x_sq_sum, "xy_sum=",xy_sum)```
```    print("m=", m, "c=", c)```
```    return m, c```
```x_data = [2,3,5,7,9]```
```y_data = [4,5,7,10,15]```
```least_squares([x_data,y_data])```

testing accuracy

```def measure_error(data1, data2):```
```    assert len(data1) == len(data2)```
```    err_total = 0```
```    for i in range(0, len(data1)):```
```        err_total = err_total + (data1[i] - data2[i]) ** 2```
```    err = math.sqrt(err_total / len(data1))```
```    return err```

```m, c = least_squares([x_data,y_data])```
```linear_data = []```
```for x in x_data:```
```    y = m * x + c```
```    linear_data.append(y)```
```print(measure_error(y_data,linear_data))```

Graphing the data

```import matplotlib.pyplot as plt```
```def make_graph(x_data, y_data, linear_data):```
```    plt.plot(x_data, y_data, label="Original Data")```
```    plt.plot(x_data, linear_data, label="Line of best fit")```
```    plt.grid()```
```    plt.legend()```
```    plt.show()```
```x_data = [2,3,5,7,9]```
```y_data = [4,5,7,10,15]]```
```m,c = least_squares([x_data,y_data])```
```linear_data = []```
```for x in x_data:```
```    y = m * x + c```
```    # add the result to the linear_data list```
```    linear_data.append(y)```
```make_graph(x_data, y_data, linear_data)```

Predicting life expectancy
Lets use real data from gapminder, download gapminder-life-expectancy.csv

Code to load the CSV file and predict 

```import pandas as pd```
```def process_life_expectancy_data(filename, country, min_date, max_date):```
```    df = pd.read_csv(filename, index_col="Life expectancy")```
```    life_expectancy = df.loc[country, str(min_date):str(max_date)]```
```    x_data = list(range(min_date, max_date + 1))```
```    m, c = least_squares([x_data, life_expectancy])```
```    linear_data = []```
```    for x in x_data:```
```        y = m * x + c```
```        linear_data.append(y)```
```    error = measure_error(life_expectancy, linear_data)```
```    print("error is ", error)```
```    make_graph(x_data, life_expectancy, linear_data)```
```process_life_expectancy_data("../data/gapminder-life-expectancy.csv", "United Kingdom", 1950, 2010)```

## Exercises

* model life expectancy for Germany 1950-2000
* predict german life expectancy 2001-2016

## Logarithmic regression

Way around linear limiations, use gapminder graphs to illustrate
logarithmis inverse of exponents. 

example code to load life expectancy and gdp

```def read_data(gdp_file, life_expectancy_file, year):```
```    df_gdp = pd.read_csv(gdp_file, index_col="Country Name")```
```    gdp = df_gdp.loc[:, year]```
```    df_life_expt = pd.read_csv(life_expectancy_file,index_col="Life expectancy")```
```    life_expectancy = df_life_expt.loc[:, year]```
```    data = []```
```    for country in life_expectancy.index:```
```        if country in gdp.index:```
```            if (math.isnan(life_expectancy[country]) is False) and (math.isnan(gdp[country]) is False):```
```                    data.append((country, life_expectancy[country],gdp[country]))```
```            else:```
```                print("Excluding ", country, ",NaN in data (life_exp = ", life_expectancy[country], "gdp=", gdp[country], ")")```
```        else:```
```            print(country, "is not in the GDP country data")```
```    combined = pd.DataFrame.from_records(data, columns=("Country","Life Expectancy", "GDP"))```
```    combined = combined.set_index("Country")```
```    # we'll need sorted data for graphing properly later on```
```    combined = combined.sort_values("Life Expectancy")```
```    return combined```

Modify process_data function to take the log of the data

add ```import math```

```gdp = data["GDP"].tolist()```
```gdp_log = data["GDP"].apply(math.log).tolist()```
```life_exp = data["Life Expectancy"].tolist()```
```m, c = least_squares([life_exp, gdp_log])```

when graphing we can choose either the log or the linear version.

``` # list for logarithmic version```
```    log_data = []```
```    # list for raw version```
```    linear_data = []```
```    for x in life_exp:```
```        y_log = m * x + c```
```        log_data.append(y_log)```
```        y = math.exp(y_log)```
```        linear_data.append(y)```
```    # uncomment for log version, further changes needed in make_graph too```
```    # make_graph(life_exp, gdp_log, log_data)```
```    make_graph(life_exp, gdp, linear_data)```


change line in least_squares function to treat data as floats, previously we had integers on the x axis for years

```  x = int(data[0][i])```

becomes

```  x = data[0][i]```

Now need a scatter graph to instead of line plot.

```def make_graph(x_data, y_data, linear_data):```
```    plt.scatter(x_data, y_data, label="Original Data")```
```    plt.plot(x_data, linear_data, color="orange", label="Line of best fit")```
```    plt.grid()```
```    plt.legend()```
```    plt.show()```

## Exercises

* compare log and linear graphs
* remove outliers from the data


# Sklearn

sklearn is a library with lots of useful ML functions.

Includes a linear regression library

```import numpy as np```
```import sklearn.linear_model as skl_lin```

replace our call to least_squares with:

```x_data_arr = np.array(x_data).reshape(-1, 1)```
```life_exp_arr = np.array(life_expectancy).reshape(-1, 1)```
```regression = skl_lin.LinearRegression().fit(x_data_arr, life_exp_arr)```
```m = regression.coef_[0][0]```
```c = regression.intercept_[0]```

computing output changes to

```linear_data = regression.predict(x_data_arr)```

test it.

Sklearn also includes error measuring code:

```import sklearn.metrics as skl_metrics```
```error = math.sqrt(skl_metrics.mean_squared_error(life_exp_arr, linear_data))```

## Exercises

* compare scikit learn and own implementation of linear regression
* predict german life expectancy

## Polynomial regression

Useful for non-linear data.

```import sklearn.preprocessing as skl_pre```

```polynomial_features = skl_pre.PolynomialFeatures(degree=5)``
```x_poly = polynomial_features.fit_transform(x_data_arr)```
```polynomial_model = skl_lin.LinearRegression().fit(x_poly, life_exp_arr)```
```polynomial_data = polynomial_model.predict(x_poly)```
```make_graph(x_data, life_expectancy, polynomial_data)```


do some predicitions:

```predictions_x = np.array(list(range(2001,2017))).reshape(-1, 1)```
```predictions_polynomial = polynomial_model.predict(polynomial_features.fit_transform(predictions_x))```
```predictions_linear = regression.predict(predictions_x)```


measure error:
```linear_error = math.sqrt(skl_metrics.mean_squared_error(life_exp_arr, linear_data))```
```print("linear error is ", linear_error)```
```polynomial_error = math.sqrt(skl_metrics.mean_squared_error(life_exp_arr, polynomial_data))```
```print("polynomial error is", polynomial_error)```

## Exercises:
* compare linear and polynomial models

# Clustering

Finds groups in data

Also used in data compresssion and pattern recognition. 

## K Means clustering

Analogy, randomly place a load of cafes in a city, see which ones are more popular, move the unpopular ones closer to the popular ones. Repeat until we have clusters of cafes in a few areas.

sklearn has a kmeans implementation although its relatively simple, we'll just stick to their version.

### advantages/Limitations of kmeans

* requires number of clusters to be known in advance, struggles on irregular or overlapping/concentric shapes.
* fast and easy to compute
* low memory overhead, suitable for large datasets
* good default option

### Exercises
* Kmeans with overlapping clsuters
* how many clusters

## Spectral clustering

works better with concentric circles. Adds extra dimensions to the data.

### Exercises
* comparing kmeans and spectral performance


# Neural Networks

Based on how the brain works. Concept of artifical neuron. Good at classification tasks, image recognition.

## Perceptrons

Multiple inputs, each multiplied by a weight. Usually scaled 0 to 1.0. 
Sum of all inputs.
Activation function for the sum. Threshold in original perceptron.

linear separability problems

### Multilayer perceptrons

solves linear separability

sklearn implementation
minst data set 

test/training data

### Exercises

* changing learning parameters
* using your own handwriting

## Cross Validation

use all the data for both training/testing. Multiple iterations. 

### Exercises

* cloud image classification 



{% include links.md %}
