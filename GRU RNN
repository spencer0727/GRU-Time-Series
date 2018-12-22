import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf

#Setting the Validation and Test ratios.
valid_set_size_percentage = 10 
test_set_size_percentage = 10 

#Defining the dataframe 
df = pd.read_csv("Appsto_close.csv", index_col = 0)

#Checking the distribution and features of the data
df.describe()

#Normalizing the data, I used the MiniMax Scaler becuase it essentially shrinks the range such that the range is now between 0 and 1 (or -1 to 1 if there are negative values).
This scaler works better for cases in which the standard scaler might not work so well. If the distribution is not Gaussian or the standard deviation is very small, the min-max scaler works better.
However, it is sensitive to outliers, so if there are outliers in the data, you might want to consider the Robust Scaler below.
