import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

def main():
    # Import the CSV files as DataFrames
    df = pd.read_csv('AASG_data_176_final_project_data_edited.csv')
    conditions = pd.read_csv('AASG_Thermed_AllThicksAndConds.csv')

    # Number of runs for each model
    num_runs = 100

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

    # call function to perform multivariate linear regression
    # Initialize a list to store the coefficient of determinations from each run of the regression. Sim. for following models.
    mvlr_mean = []
    for i in range(num_runs):
        mean, var = multivariable_linear_regression(x, y_true)
        mvlr_mean.append(mean)
    print("Multivariable linear regression with cross validation:")
    mvlr_mean_mean, mvlr_mean_variance = stat(mvlr_mean)
    print('Mean R^2: ', mvlr_mean_mean)
    print('Variance of R^2: ', mvlr_mean_variance)

    # call function to perform PCA analysis
    pca_data = PCA_analysis(x)

    # Perform multivariable linear regression analysis again after PCA analysis
    pca_mvlr_mean = []
    for i in range(num_runs):
        mean, var = multivariable_linear_regression(pca_data, y_true)
        pca_mvlr_mean.append(mean)
    print("Multivariable linear regression with cross validation for PCA data:")
    pca_mvlr_mean_mean, pca_mvlr_mean_variance = stat(pca_mvlr_mean)
    print('Mean R^2: ', pca_mvlr_mean_mean)
    print('Variance of R^2: ', pca_mvlr_mean_variance)

    nn_mean = []
    for i in range(num_runs):
        mean, var = neural_network(x, y_true)
        nn_mean.append(mean)
    print("Neural network with cross validation:")
    nn_mean_mean, nn_mean_variance = stat(nn_mean)
    print('Mean R^2: ', nn_mean_mean)
    print('Variance of R^2: ', nn_mean_variance)

    pca_nn_mean = []
    for i in range(num_runs):
        mean, var = neural_network(pca_data, y_true)
        pca_nn_mean.append(mean)
    print("Neural network with cross validation for PCA data:")
    pca_nn_mean_mean, pca_nn_mean_variance = stat(pca_nn_mean)
    print('Mean R^2: ', pca_nn_mean_mean)
    print('Variance of R^2: ', pca_nn_mean_variance)

    # Perform binary logistic regression for identifying high/low temperature wells based on variables of interest
    # Assign true labels of  high and low temperature for each datapoint and add to dataframe
    df['well performance'] = assign_labels(df['MeasuredTemperature (F)'])
    # assign variable for the labels
    temperature_labels = df['well performance']

    # run binary logistic classifier function with x array of all four variables of interest from above
    blc_accuracy_mean = []
    for i in range(num_runs):
        mean, var = binary_logistic_classifier(temperature_labels, x)
        blc_accuracy_mean.append(mean)
    print("Binary logistic classification with cross validation:")
    blc_accuracy_mean_mean, blc_accuracy_mean_variance = stat(blc_accuracy_mean)
    print('Mean accuracy: ', blc_accuracy_mean_mean)
    print('Variance accuracy: ', blc_accuracy_mean_variance)

    # perform multiclass logistic regression for determining well location in U.S. state based on well temperature
    labels = df['State']
    temperature = pd.DataFrame(df['MeasuredTemperature (F)'])

    multi_accuracy_mean = []
    for i in range(num_runs):
        mean, var = multi_class_logistic_classifier(labels, temperature)
        multi_accuracy_mean.append(mean)
    print("Binary logistic classification with cross validation:")
    multi_accuracy_mean_mean, multi_accuracy_mean_variance = stat(multi_accuracy_mean)
    print('Mean accuracy: ', multi_accuracy_mean_mean)
    print('Variance accuracy: ', multi_accuracy_mean_variance)


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
    """Performs a linear regression for multiple x variables of interest to predict well temperature. Returns the
    average coefficient of determination from the cross validation tests."""
    # creates the linear regression model
    model = LinearRegression()
    # Perform k-fold cross validation with 10 splits
    cv = KFold(n_splits=10, shuffle=True)
    scores = cross_val_score(model, x, y, cv=cv, scoring='r2')
    return np.mean(scores), np.var(scores)



def neural_network(x, y):
    """Predicts well temperature based on input variable x through neural network model. Returns the
    average coefficient of determination from the cross validation tests."""
    # build ing the neural network model using MLPRegressor function
    model = MLPRegressor(hidden_layer_sizes=(100,), activation="relu", solver='adam', max_iter=10000)
    # Perform k-fold cross validation with 10 splits
    cv = KFold(n_splits=10, shuffle=True)
    scores = cross_val_score(model, x, y, cv=cv, scoring='r2')
    return np.mean(scores), np.var(scores)


def PCA_analysis(data):
    """Performs a Principal Component Analysis for reducing the number of dimensions of the given data to three"""
    # standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    # Reduce components from 4 to 3
    pca = PCA(n_components=3)
    # fit model to standardized data
    pca_data = pca.fit_transform(scaled_data)
    # Report the explained variance ratio
    """
    print("Explained Variance Ratio: ", pca.explained_variance_ratio_)
    # plot the 3D figure after performing PCA analysis
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pca_data[:,0], pca_data[:,1], pca_data[:,2])
    ax.set_xlabel('First Principle Component')
    ax.set_ylabel('Second Principle Component')
    ax.set_zlabel('Third Principle Component')
    plt.show()
    """
    return pca_data

def assign_labels(dataframe):
    """Assigns true low temperature and high temperature labels to the dataset and returns that array of labels."""
    # create empty array to add labels to
    labels = []
    # cycle through the entire array to assign labels
    for datapoint in dataframe:
        # assign 'high temperature' label to datapoint if above 230 C and add to array
        if int(datapoint) > 302:
            labels.append('high temperature')
        # assign 'low temperature' label to datapoint if =< 230 C and add to array
        else:
            labels.append('low temperature')
    return labels


def multi_class_logistic_classifier(labels, temperature):
    """Performs a logistic regression for the multiclass label of U.S. state location. Evaluates the performance of
    the model by calculating and reporting accuracy."""
    # create multiclass logistic regression model
    logistic_clf = LogisticRegression(solver='lbfgs', max_iter=10000)
    # Perform k-fold cross validation with 10 splits
    cv = KFold(n_splits=10, shuffle=True)
    scores = cross_val_score(logistic_clf, temperature, labels, cv=cv, scoring='accuracy')
    return np.mean(scores), np.var(scores)


def binary_logistic_classifier(labels, x_variable):
    """Performs a logistic regression for the binary label of high/low temperature well based on a given
    independent variable. Evaluates the performance of the model by calculating and reporting accuracy."""
    # create multiclass logistic regression model
    logistic_clf = LogisticRegression(solver='liblinear', max_iter=10000)
    # Perform k-fold cross validation with 10 splits
    cv = KFold(n_splits=10, shuffle=True)
    scores = cross_val_score(logistic_clf, x_variable, labels, cv=cv, scoring='accuracy')
    return np.mean(scores), np.var(scores)

def stat(list):
    """Takes in a list of values and returns the mean and variance of those values."""
    mean = np.mean(list)
    variance = np.var(list)
    return mean, variance



if __name__ == '__main__':
    main()