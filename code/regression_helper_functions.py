# Import common libraries and modules
import pandas as pd # pandas is great library for storing tabular data
import matplotlib.pyplot as plt # plot module
import math # library for common math operations (log, exp)
import numpy as np # for working with numpy arrays; Sklearn functions typically take numpy arrays as input

# Import modules from Sklearn library
import sklearn.preprocessing as skl_pre # needed for polynomial regression
import sklearn.linear_model as skl_lin # linear models
import sklearn.metrics as skl_metrics # error metrics

# Type-hinting libraries
from typing import List, Tuple, Optional

# Helper functions defined below
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
        y_pred = None # FIXME: get a model prediction from each x value
        
        #add the result to the linear_data list
        linear_preds.append(y_pred)
        
    return(linear_preds)

def make_regression_graph(x_train: List[float], y_train: List[float], 
                          y_train_pred: List[float], axis_labels: Tuple[str, str], 
                          x_test: Optional[List[float]] = None, 
                          y_test: Optional[List[float]] = None,
                          y_test_pred: Optional[List[float]] = None) -> None:
    """
    Plot training data points and a model's predictions (line) on a graph. Optionally, 
    plot test data as well.

    Args:
        x_train (List[float]): A list of x-coordinates for training data points.
        y_train (List[float]): A list of corresponding y-coordinates for training data points.
        y_train_pred (List[float]): A list of predicted y-values from a model (line) for training data.
        axis_labels (Tuple[str, str]): A tuple containing the labels for the x and y axes.
        x_test (Optional[List[float]]): A list of x-coordinates for test data points (optional).
        y_test (Optional[List[float]]): A list of corresponding y-coordinates for test data points (optional).
        y_test_pred (List[float]): A list of predicted y-values from a model (line) for test data.

    Returns:
        None: The function displays the plot but does not return a value.
    """
    # Plot training data
    plt.scatter(x_train, y_train, label="Training Data")

    # Plot test data if provided
    if x_test is not None and y_test is not None:
        plt.scatter(x_test, y_test, color='orange', label="Test Data")
        
    # Concatenate x_data and y_data for both training and test sets
    x_all = x_train + (x_test if x_test is not None else [])
    
    # Concatenate y_pred for both training and test sets
    y_pred_all = y_train_pred + (y_test_pred if y_test_pred is not None else [])

    # Plot a single continuous line for all data points. Line is fit only
    # to training data.
    plt.plot(x_all, y_pred_all, label="Line of best fit")

    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.grid()
    plt.legend()

    plt.show()
    
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
        err_total = None # FIXME: add up the squared error for each observation

    err = math.sqrt(err_total / len(y))
    return err
    
def process_life_expectancy_data(
    filename: str,
    country: str,
    train_data_range: Tuple[int, int],
    test_data_range: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
    """
    Model and plot life expectancy over time for a specific country.
    
    Args:
        filename (str): The filename of the CSV data file.
        country (str): The name of the country for which life expectancy is modeled.
        train_data_range (Tuple[int, int]): A tuple representing the date range (start, end) used 
                                            for fitting the model.
        test_data_range (Optional[Tuple[int, int]]): A tuple representing the date range 
                                                     (start, end) for testing the model.
        
    Returns:
        Tuple[float, float]: A tuple containing the slope (m) and the y-intercept (c) of the 
        line of best fit.
    """

    # Extract date range used for fitting the model
    min_date_train = train_data_range[0]
    max_date_train = train_data_range[1]
    
    # Read life expectancy data
    df = pd.read_csv(filename, index_col="Life expectancy")

    # get the data used to estimate line of best fit (life expectancy for specific country across some date range)
    # we have to convert the dates to strings as pandas treats them that way
    y_train = df.loc[country, str(min_date_train):str(max_date_train)]

    # create a list with the numerical range of min_date to max_date
    # we could use the index of life_expectancy but it will be a string
    # we need numerical data
    x_train = list(range(min_date_train, max_date_train + 1))

    # calculate line of best fit
    # FIXME: Uncomment the below line of code and fill in the blank
#     m, c = _______([x_train, y_train])

    # Get model predictions for train data. 
    # FIXME: Uncomment the below line of code and fill in the blank 
#     y_train_pred = _______(x_train, m, c)
    
    # FIXME: Uncomment the below line of code and fill in the blank
#     train_error = _______(y_train, y_train_pred)

    print("Train RMSE =", format(train_error,'.5f'))
    if test_data_range is None:
        make_regression_graph(x_train, y_train, y_train_pred, ['Year', 'Life Expectancy'])
    
    # Test RMSE
    if test_data_range is not None:
        min_date_test = test_data_range[0]
        if len(test_data_range)==1:
            max_date_test=min_date_test
        else:
            max_date_test = test_data_range[1]
            
        # extract test data (x and y)
        x_test = list(range(min_date_test, max_date_test + 1))
        y_test = df.loc[country, str(min_date_test):str(max_date_test)]
        
        # get test predictions
        y_test_pred = get_model_predictions(x_test, m, c)
        
        # measure test error
        test_error = measure_error(y_test, y_test_pred)    
        print("Test RMSE =", format(test_error,'.5f'))
        
        # plot train and test data along with line of best fit 
        make_regression_graph(x_train, y_train, y_train_pred,
                              ['Year', 'Life Expectancy'], 
                              x_test, y_test, y_test_pred)

    return m, c


import pandas as pd
import math
from typing import Tuple

def read_data(gdp_file: str, life_expectancy_file: str, year: str) -> pd.DataFrame:
    """
    Read GDP and life expectancy data for a specific year for all countries with data available.
    Exclude any countries where data is missing for either GDP or life expectancy.

    Args:
        gdp_file (str): The file path to the GDP data file.
        life_expectancy_file (str): The file path to the life expectancy data file.
        year (str): The specific year for which data is requested.

    Returns:
        pd.DataFrame: A DataFrame containing the combined data for countries with available data.
    """

    # Read GDP data
    df_gdp = pd.read_csv(gdp_file, index_col="Country Name")
    gdp = df_gdp.loc[:, year]

    # Read life expectancy data
    df_life_expt = pd.read_csv(life_expectancy_file, index_col="Life expectancy")

    # Get the life expectancy for the specified year
    life_expectancy = df_life_expt.loc[:, year]

    data = []

    for country in life_expectancy.index:
        if country in gdp.index:
            # Exclude any country where data is unknown
            if not (math.isnan(life_expectancy[country]) or math.isnan(gdp[country])):
                data.append((country, life_expectancy[country], gdp[country]))
            else:
                print("Excluding", country, f"NaN in data (life_exp = {life_expectancy[country]}, GDP = {gdp[country]})")
        else:
            print(country, "is not in the GDP country data")

    # Create a DataFrame from the collected data
    combined = pd.DataFrame.from_records(data, columns=("Country", "Life Expectancy", "GDP"))
    combined = combined.set_index("Country")

    # Sort the data for proper graphing
    combined = combined.sort_values("Life Expectancy")
    
    return combined

def process_life_expt_gdp_data(gdp_file: str, life_expectancy_file: str, year: str) -> None:
    """
    Model and plot the relationship between life expectancy and log(GDP) for a specific year.

    Args:
        gdp_file (str): The file path to the GDP data file.
        life_expectancy_file (str): The file path to the life expectancy data file.
        year (str): The specific year for which data is analyzed.

    Returns:
        None: The function generates and displays plots but does not return a value.
    """
    data = read_data(gdp_file, life_expectancy_file, year)

    gdp = data["GDP"].tolist()
    # FIXME: uncomment the below line and fill in the blank
    #    log_gdp = data["GDP"].apply(____).tolist()
    
    life_exp = data["Life Expectancy"].tolist()

    m, c = least_squares([life_exp, log_gdp])

    # model predictions on transformed data
    log_gdp_preds = []
    # predictions converted back to original scale
    gdp_preds = []
    for x in life_exp:
        log_gdp_pred = m * x + c
        log_gdp_preds.append(log_gdp_pred)
        # FIXME: Uncomment the below line of code and fill in the blank
#         gdp_pred = _____(log_gdp_pred)

        gdp_preds.append(gdp_pred)

    # plot both the transformed and untransformed data
    make_regression_graph(life_exp, log_gdp, log_gdp_preds, ['Life Expectancy', 'log(GDP)'])
    make_regression_graph(life_exp, gdp, gdp_preds, ['Life Expectancy', 'GDP'])

    # typically it's best to measure error in terms of the original data scale
    train_error = measure_error(gdp_preds, gdp)
    print("Train RMSE =", format(train_error,'.5f'))
    
def process_life_expectancy_data_sklearn(filename, country, train_data_range, test_data_range=None):
    """Model and plot life expectancy over time for a specific country. Model is fit to data 
    spanning train_data_range, and tested on data spanning test_data_range"""

    # Extract date range used for fitting the model
    min_date_train = train_data_range[0]
    max_date_train = train_data_range[1]
    
    # Read life expectancy data
    df = pd.read_csv(filename, index_col="Life expectancy")

    # get the data used to estimate line of best fit (life expectancy for specific 
    # country across some date range)
    
    # we have to convert the dates to strings as pandas treats them that way
    y_train = df.loc[country, str(min_date_train):str(max_date_train)]
    
    # create a list with the numerical range of min_date to max_date
    # we could use the index of life_expectancy but it will be a string
    # we need numerical data
    x_train = list(range(min_date_train, max_date_train + 1))
    
    # NEW: Sklearn functions typically accept numpy arrays as input. This code will convert our list data into numpy arrays (N rows, 1 column)
    x_train = np.array(x_train).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)

    # OLD VERSION: m, c = least_squares([x_train, y_train])
    regression = None # FIXME: calculate line of best fit and extract m and c using sklearn. 
    
    # extract slope (m)
    # The coef_ attribute stores the coefficients (slopes) of the linear model for each feature. It is a 2D array where each row corresponds to a target variable (in the case of multi-output regression) and each column corresponds to a feature. For simple linear regression with one feature, you typically have one row and one column, so you access the slope as coef_[0][0].
    m = regression.coef_[0][0] 
    
    # extract intercept (stored separately from coef unlike some libraries)
    c = regression.intercept_[0]
    
    # print model parameters
    print("Results of linear regression:")
    print("m =", format(m,'.5f'), "c =", format(c,'.5f'))

    # OLD VERSION: y_train_pred = get_model_predictions(x_train, m, c)
    y_train_pred = None # FIXME: get model predictions for test data. 
    
    # OLD VERSION: train_error = measure_error(y_train, y_train_pred) 
    train_error = None # FIXME: calculate model train set error. 

    print("Train RMSE =", format(train_error,'.5f'))
    if test_data_range is None:
        make_regression_graph(x_train.tolist(), 
                              y_train.tolist(), 
                              y_train_pred.tolist(), 
                              ['Year', 'Life Expectancy'])
    
    # Test RMSE
    if test_data_range is not None:
        min_date_test = test_data_range[0]
        if len(test_data_range)==1:
            max_date_test=min_date_test
        else:
            max_date_test = test_data_range[1]
        x_test = list(range(min_date_test, max_date_test + 1))
        y_test = df.loc[country, str(min_date_test):str(max_date_test)]
        
        # convert data to numpy array
        x_test = np.array(x_test).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)
        
        # get predictions
        y_test_pred = regression.predict(x_test)
        
        # measure error
        test_error = math.sqrt(skl_metrics.mean_squared_error(y_test, y_test_pred))
        print("Test RMSE =", format(test_error,'.5f'))
        
        # plot train and test data along with line of best fit 
        make_regression_graph(x_train.tolist(), y_train.tolist(), y_train_pred.tolist(),
                              ['Year', 'Life Expectancy'], 
                              x_test.tolist(), y_test.tolist(), y_test_pred.tolist())

    return m, c


def process_life_expectancy_data_poly(degree: int, 
                                      filename: str, 
                                      country: str, 
                                      train_data_range: Tuple[int, int], 
                                      test_data_range: Optional[Tuple[int, int]] = None) -> None:
    """
    Model and plot life expectancy over time for a specific country using polynomial regression.

    Args:
        degree (int): The degree of the polynomial regression.
        filename (str): The CSV file containing the data.
        country (str): The name of the country for which the model is built.
        train_data_range (Tuple[int, int]): A tuple specifying the range of training data years (min_date, max_date).
        test_data_range (Optional[Tuple[int, int]]): A tuple specifying the range of test data years (min_date, max_date).

    Returns:
        None: The function displays plots but does not return a value.
    """

    # Extract date range used for fitting the model
    min_date_train = train_data_range[0]
    max_date_train = train_data_range[1]
    
    # Read life expectancy data
    df = pd.read_csv(filename, index_col="Life expectancy")

    # get the data used to estimate line of best fit (life expectancy for specific country across some date range)
    # we have to convert the dates to strings as pandas treats them that way
    y_train = df.loc[country, str(min_date_train):str(max_date_train)]
    
    # create a list with the numerical range of min_date to max_date
    # we could use the index of life_expectancy but it will be a string
    # we need numerical data
    x_train = list(range(min_date_train, max_date_train + 1))
    
    # This code will convert our list data into numpy arrays (N rows, 1 column)
    x_train = np.array(x_train).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    
    # Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, if an input sample is two dimensional and of the form [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2]. 
    # for a 5-degree polynomial applied to one feature (dates), we will get six new features: [1, x, x^2, x^3, x^4, x^5]
    polynomial_features = None # FIXME: initialize polynomial features, [1, x, x^2, x^3, ...]
    
    x_poly_train = None # FIXME:  apply polynomial transformation to training data

    print('x_train.shape', x_train.shape)
    print('x_poly_train.shape', x_poly_train.shape)

    # Calculate line of best fit using sklearn.
    regression = None # fit regression model

    # Get model predictions for test data
    y_train_pred = regression.predict(x_poly_train)
    
    # Calculate model train set error   
    train_error = math.sqrt(skl_metrics.mean_squared_error(y_train, y_train_pred))

    print("Train RMSE =", format(train_error,'.5f'))
    if test_data_range is None:
        make_regression_graph(x_train.tolist(), 
                              y_train.tolist(), 
                              y_train_pred.tolist(), 
                              ['Year', 'Life Expectancy'])
    
    # Test RMSE
    if test_data_range is not None:
        min_date_test = test_data_range[0]
        if len(test_data_range)==1:
            max_date_test=min_date_test
        else:
            max_date_test = test_data_range[1]
            
        # index data
        x_test = list(range(min_date_test, max_date_test + 1))
        y_test = df.loc[country, str(min_date_test):str(max_date_test)]
        
        # convert to numpy array 
        x_test = np.array(x_test).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)
        
        # transform x data
        x_poly_test = polynomial_features.fit_transform(x_test)
        
        # get predictions on transformed data
        y_test_pred = regression.predict(x_poly_test)
        
        # measure error
        test_error = math.sqrt(skl_metrics.mean_squared_error(y_test, y_test_pred))
        print("Test RMSE =", format(test_error,'.5f'))
        
        # plot train and test data along with line of best fit 
        make_regression_graph(x_train.tolist(), y_train.tolist(), y_train_pred.tolist(),
                              ['Year', 'Life Expectancy'], 
                              x_test.tolist(), y_test.tolist(), y_test_pred.tolist())