# CHEMENG-277-Final-Project
The CHEMENG 277 final project repository for the project "Predicting Wellbore Temperature"

BACKGROUND
Obtaining temperatures from within a geothermal well is limited by the high cost and quick degradation of measurement instruments. Additionally, well drilling depths commonly extend deeper than well casings. Since measurement instruments cannot withstand the high temperatures of the un-lined wellbore, accurate reservoir temperature measurements cannot be obtained. While numerous studies use computational and machine learning methods for predicting well temperature during drilling or operation, few studies apply machine learning techniques to predict well temperature before drilling occurs. In this study, we seek to develop a model that will predict wellbore temperature based on geothermal reservoir features such as rock conductivity, geothermal gradient, well surface temperature and well depth. Estimates of these properties could be input into the model before drilling, or after drilling an exploratory well, to predict whether the development of a geothermal plant in a particular geothermal field would be energetically favorable and economically feasible.
In this study, we use multivariable linear regression and neural networks in attempt to model geothermal well temperature given a depth, surface temperature, heat gradient, and rock conductivity. Attempting to create more accurate models, we perform an ablation study on the number of hidden layers in the neural network model, and we check the stastical significance of the multivariable linear regression's parameter coefficients to see if a parameter could be removed. After these studies, we reduced the dimensionality of the data set from four to three through PCA, and ran the multivariable linear regression and the neural network models with the new PCA dataset. We then used the original temperature values and corresponding labels in binary logistic classification to predict whether a well is high temperature, and in multiclass logistic regression to predict the U.S. state that a geothermal well is resides in.

DATA COLLECTION AND PROCESSING STEPS
Data for this work was obtained from the U.S. Department of Energy Geothermal Data Repository (GDR). Publicly available geothermal data from Cornell University titled “Appalachian Basin Play Fairway Analysis: Thermal Quality Analysis in Low-Temperature Geothermal Play Fairway Analysis (GPFA-AB)” formed the dataset for the experimental models developed. The data used for this modeling was taken from the project zip file titled “ThermalQualityAnalysisThermalModelDataFilesStateWellTemperatureDatabases”. Files titled “AASG_Processed” and “AASG_Thermed_AllThicksAndConds” were edited to only include data concerning any well type label, conductivity, true vertical depth, surface temperature, temperature gradient, heat flow, and measured temperature. After trimming the data to only relevant model information, the files were converted to .csv files for easier processing in python. The final edited files are 'AASG_data_176_final_project_data_edited.csv' and 'AASG_Thermed_AllThicksAndConds.csv' and are available in the GitHub repository in a .zip file. This file must be unzipped and added to the same folder as the python files to run correctly.

PACKAGES USED
Please see requirements.txt for all python packages used for the analysis.

RUNNING INSTRUCTIONS
There are three python files that contain code for the project: main.py, NeuralNetworkParameterStudies.py, and RegressionParameterTest.py. The main analysis is written in main.py, while the other two files serve as the nerual network ablation study and the test for linear regression coefficient statstical significance.

MAIN.PY
main.py takes in the edited datasets and outputs the mean and the variance of the metrics used to evaluate the models' performances over a defined number of runs. For the multivariable linear regression and the neural network, the metric of interest is the coefficient of determination, and for the logistic classifiers, their accuracy. Each model uses a ten K-fold cross validation train/test split, with the mean of the metric from the cross validation being considered the metric from the overall run. The number of times that each model is ran can be changed by altering the variable "num_runs" in line 17 of main.py. There are no user inputs for running the script besides running the file.

NEURALNETWORKPARAMETERSTUDIES.PY
NeuralNetworkParameterStudies.py similaraly takes in the edited data sets and then runs an ablation study comparing neural networks with 1, 2, or 3 hidden layers of 100 neurons each. Each neural network is run 5 times with the coefficients of determination averaged in order to account for the variability of the metric from run to run. If more or less datapoints are desired, the user can change the variable "num_runs" on line 27. Again, the only user action needed to run the ablation study is to run the python file.

REGRESSIONPARAMETERTEST.PY
RegressionParameterTest.py similarly takes in the edited data files and then determines the p-value for each parameter's coefficient in the regression. The regression is simply run once. No user input is needed. 

