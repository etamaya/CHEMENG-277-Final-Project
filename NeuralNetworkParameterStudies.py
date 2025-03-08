import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold

def main():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('AASG_data_176_final_project_data_edited.csv')
    # import additional csv file with corresponding data
    conditions = pd.read_csv('AASG_Thermed_AllThicksAndConds.csv')
    # define x variables
    depth = df['TrueVerticalDepth (ft)']
    surface_temp = df['SurfTemp (F)']
    heat_gradient = conditions['Gradient']
    conductivity = conditions['HeatFlow']
    # define true well temperature
    y_true = df['MeasuredTemperature (F)']
    # combine all x variables into a large dataset
    data = {'depth': depth,
            'surface temp': surface_temp,
            'heat gradient': heat_gradient,
            'conductivity': conductivity}
    x_variables = pd.DataFrame(data)
    x = x_variables[['depth', 'surface temp', 'heat gradient', 'conductivity']]

    # num_runs dictates how many times each neural network configuration is run before averaging the coefficient of determination
    num_runs = 5

    # Neural Network
    # Initialize result metric lists
    r_squared_1_layer = []
    r_squared_2_layer = []
    r_squared_3_layer = []
    layers_test = [(100,), (100,100), (100,100,100)]

    # Start ablation studies for neural network layers

    for j in range(len(layers_test)):
        for i in range(num_runs):
            r_squared = neural_network(x, y_true, layers_test[j])
            if j == 1:
                r_squared_1_layer.append(r_squared)
            if j == 2:
                r_squared_2_layer.append(r_squared)
            if j == 3:
                r_squared_3_layer.append(r_squared)

    stat(r_squared_1_layer, "1")
    stat(r_squared_2_layer, "2")
    stat(r_squared_3_layer, "3")

    print(r_squared_1_layer)
    print(r_squared_2_layer)
    print(r_squared_3_layer)

def neural_network(x, y, layers):
    """Predicts well temperature based on input variable x through neural network model"""
    # build ing the neural network model using MLPRegressor function
    model = MLPRegressor(hidden_layer_sizes=layers, activation="relu", solver='adam', max_iter=10000)
    # Perform k-fold cross validation with 10 splits
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, x, y, cv=cv, scoring='r2')
    return np.mean(scores)

def stat(list, layernum):
    mean = np.mean(list)
    median = np.median(list)
    variance = np.var(list)
    print("The R squared for ", layernum,  "layer has statistics:")
    print('Mean: ', mean)
    print('Median: ', median)
    print('Variance: ', variance)

main()