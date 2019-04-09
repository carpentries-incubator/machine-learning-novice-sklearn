---
title: "The Basics of Classification"
teaching: 0
exercises: 0
questions:
- "Key question (FIXME)"
objectives:
- "Learn how to use linear regression to produce a model from data."
- "Learn how to model non-linear data using a logarithmic."
- "Learn how to measure the error between the original data and a linear model." 
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---
FIXME



# A simple linear regression

We can graph a the two variables against each other for different countries. 
The Gapminder website will allow us to create a graph between these two parameters.

https://www.gapminder.org/tools/#$state$time$value=2017&showForecast:true&delay:206.4516129032258;&entities$filter$;&dim=geo;&marker$axis_x$which=life_expectancy_years&domainMin:null&domainMax:null&zoomedMin:45&zoomedMax:84.17&scaleType=linear&spaceRef:null;&axis_y$which=gdppercapita_us_inflation_adjusted&domainMin:null&domainMax:null&zoomedMin:115.79&zoomedMax:144246.37&spaceRef:null;&size$domainMin:null&domainMax:null&extent@:0.022083333333333333&:0.4083333333333333;;&color$which=world_6region;;;&chart-type=bubbles


The relationship between these two variables clearly isn't linear. But gapminder lets us adjust the Y axis from linear to logarithmic. Click on the arrow on the left next to GDP/capita. Choose log. 

https://www.gapminder.org/tools/#$state$time$value=2017&showForecast:true&delay:206.4516129032258;&entities$filter$;&dim=geo;&marker$axis_x$which=life_expectancy_years&domainMin:null&domainMax:null&zoomedMin:45&zoomedMax:84.17&scaleType=linear&spaceRef:null;&axis_y$which=gdppercapita_us_inflation_adjusted&domainMin:null&domainMax:null&zoomedMin:115.79&zoomedMax:144246.37&scaleType=log&spaceRef:null;&size$domainMin:null&domainMax:null&extent@:0.022083333333333333&:0.4083333333333333;;&color$which=world_6region;;;&chart-type=bubbles


We can now see what appears to be a linear relationship between the two variables and could create a linear equation to predict life expectancy given the log of the GDP. We can do this using a method called [linear regression or least square regression](https://www.mathsisfun.com/data/least-squares-regression.html). 

Instead of doing this manually we could get a computer to work out the relationship (we'll do this for real later on). Getting a computer to automatically predict one variable from another (or a set of others) is the basis of all machine learning. 

## Coding a linear regression with Python 
This code will calculate 

~~~
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
~~~
{: .python}

Lets test our code by using the example data from the mathsisfun link above. 

~~~
x_data = [2,3,5,7,9]
y_data = [4,5,7,10,15]]
least_squares([x_data,y_data])
~~~
{: .python}

We should get the following results:

~~~
Results of linear regression:
x_sum= 26 y_sum= 41 x_sq_sum= 168 xy_sum= 263
m= 1.5182926829268293 c= 0.30487804878048763
~~~

### Testing the accuracy of a linear regression model

We now have a simple linear model for some data. It would be useful to test how accurate that model is. We can do this by computing the y value for every x value used in our original data and comparing the model's y value with the original. We can turn this into a single overall error number by calculating the root mean square (RMS), this squares each comparison, takes the sum of all of them, divides this by the number of items and finally takes the square root of that value. By squaring and square rooting the values we prevent negative errors from cancelling out positive ones. The RMS gives us an overall error number which we can then use to measure our model's accuracy with. The following code calculates RMS in Python. 

~~~
def measure_error(data1, data2):
    assert len(data1) == len(data2)
    err_total = 0
    for i in range(0, len(data1)):
        err_total = err_total + (data1[i] - data2[i]) ** 2

    err = math.sqrt(err_total / len(data1))
    return err
~~~
{: .python}


To calculate the RMS for the test data we just used we need to calculate the y coordinate for every x coordinate (2,3,5,7,9) that we had in the original data. 

~~~
# get the m and c values from the least_squares function
m, c = least_squares([x_data,y_data])

# create an empty list for the model y data
y_model = []

for x in x_data:
    y = m * x + c
    # add the result to the y_model list
    y_model.append(y)

# calculate the error
print(measure_error(y_data,y_model))
~~~
{: .python}

This will output an error of 0.7986268703523449, which means that on average the difference between our model and the real values is 0.7986268703523449. The less linear the data is the bigger this number will be. If the model perfectly matches the data then the value will be zero.


### Graphing the data

To compare our model and data lets graph both of them using matplotlib.

~~~
import matplotlib as plt

def make_graph(x_data, y_data, y_model):

    plt.plot(x_data, y_data, label="Original Data")
    plt.plot(x_data, y_model, label="Line of best fit")

    plt.grid()
    plt.legend()

    plt.show()
    
x_data = [2,3,5,7,9]
y_data = [4,5,7,10,15]]
m,c = least_squares([x_data,y_data])

y_model = []

for x in x_data:
    y = m * x + c
    # add the result to the y_model list
    y_model.append(y)
    
make_graph(x_data, y_data, y_model)
~~~
{: .python}




### Predicting life expectancy

~~~
# put this line at the top of the file
import pandas as pd 

def process_life_expectancy_data(filename, country, min_date, max_date):

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
    for x in x_data:
        y = m * x + c
        y_model.append(y)

    error = measure_error(life_expectancy, y_model)
    print("error is ", error)

    make_graph(x_data, life_expectancy, y_model)

process_life_expectancy_data("../data/gapminder-life-expectancy.csv",
                             "United Kingdom", 1950, 2010)
~~~
{: .python}


> # Modelling Life Expectancy
>
> Combine all the code above into a single Python file, save it into a directory called code. 
>
> In the parent directory create another directory called data
>
> Download the file https://scw-aberystwyth.github.io/machine-learning-novice/data/gapminder-life-expectancy.csv into the data directory 
>
> If you're using a Unix or Unix like environment the following commands will do this in your home directory:
>
> ~~~
> cd ~
> mkdir code
> mkdir data
> cd data
> wget https://scw-aberystwyth.github.io/machine-learning-novice/data/gapminder-life-expectancy.csv
> ~~~
> {: .bash}
>
> Adjust the program to calculate the life expectancy for Germany between 1950 and 2000. What are the values (m and c) of linear equation linking date and life expectancy?
> > ## Solution
> > ~~~
> > process_life_expectancy_data("../data/gapminder-life-expectancy.csv", "Germany", 1950, 2000)
> > ~~~ 
> > {: .python}
> > 
> > m= 0.212219909502 c= -346.784909502
> {: .solution}
{: .challenge}


> # Predicting Life Expectancy
> Use the linear equation you've just created to predict life expectancy in Germany for every year between 2001 and 2016. How accurate are your answers?
> If you worked for a pension scheme would you trust your answers to predict the future costs for paying pensioners?
> > ## Solution
> > ~~~
> > for x in range(2001,2017):
> >     print(x,0.212219909502 * x - 346.784909502)
> > ~~~
> > {: .python}
> > 
> > Predicted answers:
> > ~~~
> > 2001 77.86712941150199
> > 2002 78.07934932100403
> > 2003 78.29156923050601
> > 2004 78.503789140008
> > 2005 78.71600904951003
> > 2006 78.92822895901202
> > 2007 79.140448868514
> > 2008 79.35266877801604
> > 2009 79.56488868751802
> > 2010 79.77710859702
> > 2011 79.98932850652199
> > 2012 80.20154841602402
> > 2013 80.41376832552601
> > 2014 80.62598823502799
> > 2015 80.83820814453003
> > 2016 81.05042805403201
> > ~~~
> > Compare with the real values:
> > ~~~
> > df = pd.read_csv('../data/gapminder-life-expectancy.csv',index_col="Life expectancy")
> > for x in range(2001,2017):
> >     y = 0.215621719457 * x - 351.935837103
> >     real = df.loc['Germany', str(x)]
> >     print(x, "Predicted", y, "Real", real, "Difference", y-real)
> > ~~~
> > {: .python}
> > 
> > ~~~
> > 2001 Predicted 77.86712941150199 Real 78.4 Difference -0.532870588498
> > 2002 Predicted 78.07934932100403 Real 78.6 Difference -0.520650678996
> > 2003 Predicted 78.29156923050601 Real 78.8 Difference -0.508430769494
> > 2004 Predicted 78.503789140008 Real 79.2 Difference -0.696210859992
> > 2005 Predicted 78.71600904951003 Real 79.4 Difference -0.68399095049
> > 2006 Predicted 78.92822895901202 Real 79.7 Difference -0.771771040988
> > 2007 Predicted 79.140448868514 Real 79.9 Difference -0.759551131486
> > 2008 Predicted 79.35266877801604 Real 80.0 Difference -0.647331221984
> > 2009 Predicted 79.56488868751802 Real 80.1 Difference -0.535111312482
> > 2010 Predicted 79.77710859702 Real 80.3 Difference -0.52289140298
> > 2011 Predicted 79.98932850652199 Real 80.5 Difference -0.510671493478
> > 2012 Predicted 80.20154841602402 Real 80.6 Difference -0.398451583976
> > 2013 Predicted 80.41376832552601 Real 80.7 Difference -0.286231674474
> > 2014 Predicted 80.62598823502799 Real 80.7 Difference -0.074011764972
> > 2015 Predicted 80.83820814453003 Real 80.8 Difference 0.03820814453
> > 2016 Predicted 81.05042805403201 Real 80.9 Difference 0.150428054032
> > ~~~
> > Answers are between 0.15 years over and 0.77 years under the reality. 
> > If this was being used in a pension scheme it might lead to a slight under prediction of life expectancy and cost the pension scheme a little more than expected.
> {: .solution}
{: .challenge}




> # Predicting Historical Life Expectancy
> 
> Now change your program to measure life expectancy in Canada between 1890 and 1914. Use the resulting m and c values to predict life expectancy in 1918. How accurate is your answer?
> If your answer was inaccurate, why was it inaccurate? What does this tell you about extrapolating models like this?
> > ## Solution
> > ~~~
> > process_life_expectancy_data("../data/gapminder-life-expectancy.csv", "Canada", 1890, 1914)
> > ~~~
> > {: .python}
> > 
> > m = 0.369807692308 c = -654.215830769
> > ~~~
> > print(1918 * 0.369807692308  -654.215830769)
> > ~~~
> > {: .python}
> > predicted age: 55.0753, actual 47.17
> > Inaccurate due to WW1 and flu epedemic. Major events can produce trends that we've not seen before (or not for a long time), our models struggle to take account of things they've never seen.
> > Even if we look back to 1800, the earliest date we have data for we never see a sudden drop in life expectancy like the 1918 one.
> {: .solution}
{: .challenge}



