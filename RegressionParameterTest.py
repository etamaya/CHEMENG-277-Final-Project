import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import statsmodels.api as sm


def main():
    # Import the CSV files as DataFrames
    df = pd.read_csv('AASG_data_176_final_project_data_edited.csv')
    conditions = pd.read_csv('AASG_Thermed_AllThicksAndConds.csv')

    # Perform multivariable linear regression analysis for predicting well temperature
    # define x variables
    depth = df['TrueVerticalDepth (ft)']
    surface_temp = df['SurfTemp (F)']
    heat_gradient = conditions['Gradient']
    conductivity = conditions['HeatFlow']
    # call function to combine independent x variables
    x = create_regression_x_data(depth, surface_temp, heat_gradient, conductivity)
    # define true well temperature
    y_true = df['MeasuredTemperature (F)']

    mean, var, p_values = multivariable_linear_regression(x, y_true)
    print('P-value for regression coefficients:')
    print('y-int, depth, surface_temp, heat_gradient, conductivity')
    print(p_values)

def create_regression_x_data(depth, surface_temp, heat_gradient, conductivity):
    """Combines all independent x variables into a single pandas dataframe for multivariable
    linear regression analysis"""
    # combine all data into a large dataset
    data = {'depth': depth,
            'surface temp': surface_temp,
            'heat gradient': heat_gradient,
            'conductivity': conductivity}
    # convert to dataframe
    x_variables = pd.DataFrame(data)
    # define final combined x array
    x = x_variables[['depth', 'surface temp', 'heat gradient', 'conductivity']]
    return x


def multivariable_linear_regression(x, y):
    """Performs a linear regression for multiple x variables of interest to predict well temperature"""
    # creates the linear regression model
    model = LinearRegression()

    # Use statsmodels to get p-values for individual regression coefficients
    model.fit(x, y)
    x_with_intercept = sm.add_constant(x)
    sm_model = sm.OLS(y, x_with_intercept).fit()
    p_values = sm_model.pvalues

    # Perform k-fold cross validation with 10 splits
    cv = KFold(n_splits=10, shuffle=True)
    scores = cross_val_score(model, x, y, cv=cv, scoring='r2')
    return np.mean(scores), np.var(scores), p_values


def stat(list):
    """Takes in a list of values and returns the mean and variance of those values."""
    mean = np.mean(list)
    variance = np.var(list)
    return mean, variance


if __name__ == '__main__':
    main()