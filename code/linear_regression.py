#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:05:11 2019

@author: colin
"""

import pandas as pd
import matplotlib.pyplot as plt
import math


def least_squares(data):
    '''Calculate the least squares (linear) regression for a data set
    the data should be a single list containing two sublists, the first sublist
    should be the x data and the second the y data'''

    x_sum = 0
    y_sum = 0
    x_sq_sum = 0
    xy_sum = 0

    # the list of data should have two equal length columns
    assert len(data[0]) == len(data[1])
    assert len(data) == 2

    n = len(data[0])
    # least squares regression calculation
    for i in range(0, n):
        x = int(data[0][i])
        y = data[1][i]
        x_sum = x_sum + x
        y_sum = y_sum + y
        x_sq_sum = x_sq_sum + (x**2)
        xy_sum = xy_sum + (x*y)

    m = ((n * xy_sum) - (x_sum * y_sum))
    m = m / ((n * x_sq_sum) - (x_sum ** 2))
    c = (y_sum - m * x_sum) / n

    print("Results of linear regression:")
    print("x_sum=", x_sum, "y_sum=", y_sum, "x_sq_sum=", x_sq_sum, "xy_sum=",
          xy_sum)
    print("m=", m, "c=", c)

    return m, c


def measure_error(data1, data2):
    '''Measure the RMS error between data1 and data2'''
    assert len(data1) == len(data2)
    err_total = 0
    for i in range(0, len(data1)):
        err_total = err_total + (data1[i] - data2[i]) ** 2

    err = math.sqrt(err_total / len(data1))
    return err


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

    # calculate line of best fit
    m, c = least_squares([x_data, life_expectancy])

    y_model = []
    # calculate the y coordinates for every point in the list x
    for x in x_data:
        y = m * x + c
        # add the result to the y_model list
        y_model.append(y)

    error = measure_error(life_expectancy, y_model)
    print("error is ", error)

    make_graph(x_data, life_expectancy, y_model)


process_life_expectancy_data("../data/gapminder-life-expectancy.csv",
                             "Canada", 1800, 1920)
