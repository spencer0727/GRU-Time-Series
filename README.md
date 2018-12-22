# GRU-Time-Series
GRU Neural Network Models for Market Movement Prediction
The first model is a time series MV regression model which takes a split of 80%/10%/10% training, validiation, and testing set and the second model is a MV binary classifier. 
I had less than wonderful results with the MV regression model, however the binary classifier was fairly good for predicting exact values of daily returns, but gives satisfactory results when used to predict the direction of the movement.

The results of the MV regresion model were meausured by taking the MSE of the training data and the results with the MV binary classifier were measured by taking the F1 Score. 

The following dependencies were installed and used, Python 3.7

numpy==1.14.2 
pandas==0.22.0 
plotly==2.5.0 
scikit-learn==0.19.1 
scipy==1.0.0 
seaborn==0.8.1 
sklearn==0.0 
tensorflow==1.6.0 
tensorflow-gpu==1.8.0 
