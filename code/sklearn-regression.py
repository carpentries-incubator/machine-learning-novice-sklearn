#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:44:31 2019

@author: colin
"""
import pandas as pd
import matplotlib.pyplot as plt
import math
import sklearn.linear_model as skl_lin
import sklearn.metrics as skl_metrics
import numpy as np
import sklearn.preprocessing as skl_pre
import sklearn.pipeline as skl_pipe


def make_graph(x_data, y_data, y_model, isotonic_model):

    # rotate the x labels to fit better
    plt.xticks(rotation=90)

    # calculate the minimum and maximum life expectancy
    # floor rounds down, ceil rounds up
    min_y = math.floor(min(y_data))
    max_y = math.ceil(max(y_data))

    # evenly space y axis, interval of 1, between the min and max life exp
    plt.yticks(list(range(min_y, max_y, 1)))

    plt.xlim(min(x_data), max(x_data))
    plt.ylim(min_y, max_y)

    # draw the line
    plt.plot(x_data, y_data, label="Original Data")
    plt.plot(x_data, y_model, label="Line of best fit")
    plt.plot(x_data, isotonic_model, label="Polynomial model")

    plt.grid()

    # enable the legend
    plt.legend(loc='best')

    # draw the graph
    plt.show()


def process_life_expectancy_data(filename, country, min_date, max_date):
    '''Graph the data from a life expectancy file
    calculate the line of best fit and graph it too
    also calculate the error between the line of best fit and the data'''

    df = pd.read_csv(filename, index_col="Life expectancy")

    # get the life expectancy for the specified country/dates
    # we have to convert the dates to strings as pandas treats them that way
    life_expectancy = df.loc[country, str(min_date):str(max_date)]

    # create a list with the numerical range of min_date to max_date
    # we could use the index of life_expectancy but it will be a string
    # we need numerical data
    x_data = list(range(min_date, max_date + 1))

    # convert x_data and life_expectancy to numpy arrays
    x_data_arr = np.array(x_data).reshape(-1, 1)
    life_exp_arr = np.array(life_expectancy).reshape(-1, 1)

    # perform the lienar regression
    regression = skl_lin.LinearRegression().fit(x_data_arr, life_exp_arr)
    print(type(x_data),type(life_expectancy.tolist()))
    #iso = ir.IsotonicRegression()
    #isotonic_model = iso.fit(x_data, life_expectancy.tolist())
    #predictions_isotonic = iso.predict(list=(range(1999,1999.5)))
    predictions_x = np.array(list(range(2001,2016)))
    predictions_linear = regression.predict(predictions_x.reshape(-1,1))

    polynomial_features = skl_pre.PolynomialFeatures(degree=10)
    x_poly = polynomial_features.fit_transform(x_data_arr)
    polynomial_model = skl_lin.LinearRegression().fit(x_poly, life_exp_arr)
    polynomial_data = polynomial_model.predict(x_poly)
    
    predictions_polynomial = polynomial_model.predict(
                             polynomial_features.fit_transform(predictions_x.reshape(-1,1)))

    #polynomial_model = skl_pipe.make_pipeline(skl_pre.PolynomialFeatures(degree = 8),skl_lin.LinearRegression())
    #polynomial_model.fit(x_data_arr,life_exp_arr)

    print("polynomial predicitons",predictions_polynomial)
    print("linear predcitions",predictions_linear)
    print("actual values",df.loc[country, "2001":"2016"])

    # get the parameters from the regression
    m = regression.coef_[0][0]
    c = regression.intercept_[0]

    print("m =", m, "c=", c)

    # generate the line of best fit from our model
    y_model = regression.predict(x_data_arr)
    #isotonic_model = iso.predict(x_data)

    # calcualte the root mean squared error
    error = math.sqrt(skl_metrics.mean_squared_error(life_exp_arr, y_model))
    print("error is ", error)

    polynomial_error = math.sqrt(skl_metrics.mean_squared_error(life_exp_arr, polynomial_data))
    print("polynomial error is", polynomial_error)


    make_graph(x_data, life_expectancy, y_model, polynomial_data)


process_life_expectancy_data("../data/gapminder-life-expectancy.csv",
                             "United Kingdom", 1960, 2000)
