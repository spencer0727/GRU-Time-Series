# GRU-Time-Series Description & Prediction

# Instructions to run
The following dependencies are listed at the bottom. Must first intialize a TF environment in order to use TF. 
Please see https://www.tensorflow.org/install/gpu for help with TF installation.
Please read GRU time series for more detailed description


# GRU Neural Network Models for Market Movement Prediction

This is my first attempt for my masters the full thesis shall be published circa June 2018 

# Introduction
The first model is a time series MV regression model which takes a split of 80%/10%/10% training, validiation, and testing set and the second model is a MV binary classifier. GRU's have been shown to perform better than LSTM cells with when there is not much data available, thus I wanted to see if it made a difference. The validation set was used in order to ensure the model did not overfit the data before testing the model. 


# Results
I had less than wonderful results with the MV regression model, however the binary classifier was fairly good for predicting exact values of daily returns, but gives satisfactory results when used to predict the direction of the movement.
I ambiguously chose APPL for the financial time series, however the actual selection does not matter as much. 

The classifier is currently a work in progress as I will still need to fit the specifications properly first.


The results of the MV regresion model were meausured by taking the MSE of the training data and the results with the MV binary classifier were measured by taking the F1 Score. 

# Dependencies
Python == 3.7
numpy==1.14.2 
pandas==0.22.0 
plotly==2.5.0 
scikit-learn==0.19.1 
scipy==1.0.0 
seaborn==0.8.1 
sklearn==0.0 
tensorflow==1.6.0 
tensorflow-gpu==1.8.0 

Other Needed Downloads:
CUDA toolkit 9.0
cuDNN SDK 7.2

Specs of machine used:
NVIDIA GTX 1070
Intel CPU i7-8750H
16 GB of 2666 MHz DDR4 SDRAM


# License

MIT License

Copyright (c) 2018 Spencer Frebel

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

