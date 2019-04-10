#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:44:31 2019

@author: colin
"""
import pandas as pd
import matplotlib.pyplot as plt
import math
import sklearn.linear_model as skl
import sklearn.metrics as metrics
import numpy as np


def make_graph(x_data, y_data, y_model):

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

    plt.grid()

    # enable the legend
    plt.legend()

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
    regression = skl.LinearRegression().fit(x_data_arr, life_exp_arr)

    # get the parameters from the regression
    m = regression.coef_[0][0]
    c = regression.intercept_[0]

    print("m =", m, "c=", c)

    # generate the line of best fit from our model
    y_model = regression.predict(x_data_arr)

    # calcualte the root mean squared error
    error = math.sqrt(metrics.mean_squared_error(life_exp_arr, y_model))
    print("error is ", error)

    make_graph(x_data, life_expectancy, y_model)


process_life_expectancy_data("../data/gapminder-life-expectancy.csv",
                             "Canada", 1800, 2000)
