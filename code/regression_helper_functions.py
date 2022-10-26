# Import common libraries and modules
import pandas as pd # pandas is great library for storing tabular data
import matplotlib.pyplot as plt # plot module
import math # library for common math operations (log, exp)
import numpy as np # for working with numpy arrays; Sklearn functions typically take numpy arrays as input

# Import modules from Sklearn library
import sklearn.preprocessing as skl_pre # needed for polynomial regression

# Helper functions defined below
def least_squares(data):
    """Calculate the line of best fit for a data matrix of [x_values,y_values] using ordinary least squares optimization."""
    
    x_sum = 0
    y_sum = 0
    x_sq_sum = 0
    xy_sum = 0

    # the list of data should have two equal length columns
    assert len(data) == 2
    assert len(data[0]) == len(data[1])

    n = len(data[0])
    # least squares regression calculation
    for i in range(0, n):
        if isinstance(data[0][i],str):
            x = int(data[0][i]) # convert date string to int
        else:
            x = data[0][i] # for GDP vs life-expect data
        y = data[1][i]
        x_sum = x_sum + x
        y_sum = y_sum + y
        x_sq_sum = x_sq_sum + (x**2)
        xy_sum = xy_sum + (x*y)

    m = ((n * xy_sum) - (x_sum * y_sum))
    m = m / ((n * x_sq_sum) - (x_sum ** 2))
    c = (y_sum - m * x_sum) / n

    print("Results of linear regression:")
    print("m =", format(m,'.5f'), "c =", format(c,'.5f'))

    return m, c

def get_model_predictions(x_data, m, c):
    """Using the input slope (m) and y-intercept (c), calculate linear model predictions (y-values) for a given list of x-coordinates."""
    
    linear_preds = []
    for x in x_data:
        # FIXME: Uncomment below line and complete the line of code to get a model prediction from each x value
#         y = _______
        
        #add the result to the linear_data list
        linear_preds.append(y)
    return(linear_preds)

def make_regression_graph(x_data, y_data, y_pred, axis_labels):
    """Plot data and model predictions (line)."""

    plt.scatter(x_data, y_data, label="Data")
    plt.plot(x_data, y_pred, label="Line of best fit")
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.grid()
    plt.legend()

    plt.show()
    
# 
def measure_error(data,preds):
    """Calculating RMSE (root mean square error) of model."""
    assert len(data)==len(preds)
    err_total = 0
    for i in range(0,len(data)):
        # FIXME: Uncomment the below line and fill in the blank to add up the squared error for each observation
#         err_total = err_total + ________

    err = math.sqrt(err_total / len(data))
    return err
    
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

    # Get model predictions for test data. 
    # FIXME: Uncomment the below line of code and fill in the blank 
#     y_preds_train = _______(x_data_train, m, c)
    
    # FIXME: Uncomment the below line of code and fill in the blank
#     train_error = _______(y_data_train, y_preds_train)

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


def read_data(gdp_file, life_expectancy_file, year):
    """Return GDP and life expectancy data for a specific year (all countries with data available)"""
        
    df_gdp = pd.read_csv(gdp_file, index_col="Country Name")
    gdp = df_gdp.loc[:, year]

    df_life_expt = pd.read_csv(life_expectancy_file,
                               index_col="Life expectancy")

    # get the life expectancy for the specified country/dates
    # we have to convert the dates to strings as pandas treats them that way
    life_expectancy = df_life_expt.loc[:, year]

    data = []
    for country in life_expectancy.index:
        if country in gdp.index:
            # exclude any country where data is unknown
            if (math.isnan(life_expectancy[country]) is False) and \
               (math.isnan(gdp[country]) is False):
                    data.append((country, life_expectancy[country],
                                 gdp[country]))
            else:
                print("Excluding ", country, ",NaN in data (life_exp = ",
                      life_expectancy[country], "gdp =", gdp[country], ")")
        else:
            print(country, "is not in the GDP country data")

    combined = pd.DataFrame.from_records(data, columns=("Country",
                                         "Life Expectancy", "GDP"))
    combined = combined.set_index("Country")
    # we'll need sorted data for graphing properly later on
    combined = combined.sort_values("Life Expectancy")
    return combined

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
        gdp_preds_transformed.append(y_pred)

    # plot both the transformed and untransformed data
    make_regression_graph(life_exp, gdp_log, gdp_preds, ['Life Expectancy', 'log(GDP)'])
    make_regression_graph(life_exp, gdp, gdp_preds_transformed, ['Life Expectancy', 'GDP'])

    train_error = measure_error(gdp_preds, gdp)
    print("Train RMSE =", format(train_error,'.5f'))
    
    
def process_life_expectancy_data_sklearn(filename, country, train_data_range, test_data_range=None):
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
    
    # NEW: Sklearn functions typically accept numpy arrays as input. This code will convert our list data into numpy arrays (N rows, 1 column)
    x_data_train = np.array(x_data_train).reshape(-1, 1)
    y_data_train = np.array(y_data_train).reshape(-1, 1)

    # FIXME: calculate line of best fit and extract m and c using sklearn. OLD VERSION: m, c = least_squares([x_data_train, y_data_train])

    
    # print model parameters
    print("Results of linear regression:")
    print("m =", format(m,'.5f'), "c =", format(c,'.5f'))

    # FIXME: get model predictions for test data. OLD VERSION: y_preds_train = get_model_predictions(x_data_train, m, c)
    
    # FIXME: calculate model train set error. OLD VERSION: train_error = measure_error(y_data_train, y_preds_train)    
    train_error = math.sqrt(skl_metrics.mean_squared_error(y_data_train, y_preds_train))

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
        
        x_data_test = np.array(x_data_test).reshape(-1, 1)
        y_data_test = np.array(y_data_test).reshape(-1, 1)
        
        y_preds_test = regression.predict(x_data_test)
        test_error = math.sqrt(skl_metrics.mean_squared_error(y_data_test, y_preds_test))
        print("Test RMSE =", format(test_error,'.5f'))
        make_regression_graph(np.concatenate((x_data_train, x_data_test), axis=0), 
                              np.concatenate((y_data_train, y_data_test), axis=0), 
                              np.concatenate((y_preds_train, y_preds_test), axis=0), 
                              ['Year', 'Life Expectancy'])

    return m, c


def process_life_expectancy_data_poly(degree, filename, country, train_data_range, test_data_range=None):
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
    
    # This code will convert our list data into numpy arrays (N rows, 1 column)
    x_data_train = np.array(x_data_train).reshape(-1, 1)
    y_data_train = np.array(y_data_train).reshape(-1, 1)
    
    # Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, if an input sample is two dimensional and of the form [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2]. 
    # for a 5-degree polynomial applied to one feature (dates), we will get six new features: [1, x, x^2, x^3, x^4, x^5]
    polynomial_features = skl_pre.PolynomialFeatures(degree=degree)
    x_poly_train = polynomial_features.fit_transform(x_data_train)        
    print('x_data_train.shape', x_data_train.shape)
    print('x_poly_train.shape', x_poly_train.shape)

    # Calculate line of best fit using sklearn.
    regression = skl_lin.LinearRegression().fit(x_poly_train, y_data_train)  

    # Get model predictions for test data
    y_preds_train = regression.predict(x_poly_train)
    
    # Calculate model train set error   
    train_error = math.sqrt(skl_metrics.mean_squared_error(y_data_train, y_preds_train))

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

        x_data_test = np.array(x_data_test).reshape(-1, 1)
        y_data_test = np.array(y_data_test).reshape(-1, 1)
        
        x_poly_test = polynomial_features.fit_transform(x_data_test)
        y_preds_test = regression.predict(x_poly_test)
        test_error = math.sqrt(skl_metrics.mean_squared_error(y_data_test, y_preds_test))
        print("Test RMSE =", format(test_error,'.5f'))
        make_regression_graph(np.concatenate((x_data_train, x_data_test), axis=0), 
                              np.concatenate((y_data_train, y_data_test), axis=0), 
                              np.concatenate((y_preds_train, y_preds_test), axis=0), 
                              ['Year', 'Life Expectancy'])