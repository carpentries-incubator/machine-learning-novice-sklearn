#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:02:25 2019

@author: colin
"""

import matplotlib.pyplot as plt

def least_squares(data):
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

def make_graph(x_data, y_data, linear_data):

    plt.plot(x_data, y_data, label="Original Data")
    plt.plot(x_data, linear_data, label="Line of best fit")

    plt.grid()
    plt.legend()

    plt.show()

x_data = [2,3,5,7,9]
y_data = [4,5,7,10,15]
m,c = least_squares([x_data,y_data])

linear_data = []

for x in x_data:
    y = m * x + c
    # add the result to the linear_data list
    linear_data.append(y)

make_graph(x_data, y_data, linear_data)