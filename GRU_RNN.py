import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf

#Setting the Validation and Test ratios.
valid_set_size_percentage = 10 
test_set_size_percentage = 10 

#Defining the dataframe 
df = pd.read_csv("", index_col = 0)

#Checking the distribution and features of the data
df.describe()

'''Normalization of the data may in some instances sacrifice the accuracy of the training data in order to better generalize the testing data.
There are a few different methods which can be used, in instances where the robustness of the data is needed I will use the MinMaxScaler.
Normalizing the data, I used the MiniMax Scaler, shrinks the range such that the range is now between 0 and 1 (or -1 to 1 if there are negative values).
This scaler works better for cases in which the standard scaler might not work so well. If the distribution is not Gaussian or the standard deviation is very small, the min-max scaler works better.
However, it is sensitive to outliers, so if there are outliers in the data, you might want to consider the Robust Scaler below.'''

def normalize_data(df):
    min_max = sklearn.preprocessing.MinMaxScaler()
    df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1,1))
    return df
  
#Next we need to load the data and parse it into train, valid and test sets.
def load_data(stock, seq_len):
    data_raw = stock.as_matrix() 
    data = []
    for index in range(len(data_raw)-seq_len):
            data.append(data_raw[index: index + seq_len])
    #Separating the data into their respective groups, play around with the test size if needed.         
    data = np.array(data);
    valid_set_size = int(np.round(valid_set_size_percentage/100*data.shape[0]));  
    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]));
    train_set_size = data.shape[0] - (valid_set_size + test_set_size);
    
    #Separating the datasets
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]
    y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]
    
    x_test = data[train_set_size+valid_set_size:,:-1,:]
    y_test = data[train_set_size+valid_set_size:,-1,:]
    
    return [x_train, y_train, x_valid, y_valid, x_test, y_test]
  
#Creating a copy of the data to pass through the parser
df_stock = df.copy()
df.columns
cols = list(df_stock.columns.values)
print('df_stock.columns.values = ', cols)
df_stock_norm = df_stock.copy()
df_stock_norm = normalize_data(df_stock_norm)
seq_len = 12 

#Sep out the data
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df_stock_norm, seq_len)

#Checking the shape of each of the data 
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid.shape = ',x_valid.shape)
print('y_valid.shape = ', y_valid.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ',y_test.shape)
  
#Indexing the epoch
index_in_epoch = 0;
perm_array  = np.arange(x_train.shape[0])
np.random.shuffle(perm_array)
  
#Setting the parameters to retrive the next batch
def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array
    start = index_in_epoch
    index_in_epoch += batch_size
    
   
    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array) 
        start = 0 
        index_in_epoch = batch_size
        
    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]
 

#Setting the hyperparameter options
n_steps = seq_len-1 
n_inputs = 1 
n_neurons = 200 
n_outputs = 1
n_layers = 2
learning_rate = 0.001
batch_size = 50
n_epochs = 100 
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]
  
#Begin the processes for tensorflow 
tf.reset_default_graph()

#Set the placeholders, please note the convention of the naming, X is the matrix and y is the vector
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

#Define the layer of the RNN, I also had good results with RNN leaky activation function, and intitialize the TF train process
layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.elu)
          for layer in range(n_layers)]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)

rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
  
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons]) 
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
outputs = outputs[:,n_steps-1,:]

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
training_op = optimizer.minimize(loss)
  
with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    for iteration in range(int(n_epochs*train_set_size/batch_size)):
        x_batch, y_batch = get_next_batch(batch_size) # fetch the next training batch 
        sess.run(training_op, feed_dict={X: x_batch, y: y_batch}) 
        if iteration % int(5*train_set_size/batch_size) == 0:
            mse_train = loss.eval(feed_dict={X: x_train, y: y_train}) 
            mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid}) 
            print('%.2f epochs: Mean Square Error Training Set & Validation Set = %.6f/%.6f'%(
                iteration*batch_size/train_set_size, mse_train, mse_valid))

    y_train_pred = sess.run(outputs, feed_dict={X: x_train})
    y_valid_pred = sess.run(outputs, feed_dict={X: x_valid})
    y_test_pred = sess.run(outputs, feed_dict={X: x_test})  
  
  
#Check the MSE of the testing data for results 
mse = mean_squared_error(y_test, y_test_pred)
print('MSE: %f' % mse)
  
  
  
#Checking the direction of each of the precidtion of each moves 
corr_price_development_test = np.sum(np.equal(np.sign(y_test[:,0]-y_test[:,0]),
            np.sign(y_test_pred[:,0]-y_test_pred[:,0])).astype(int)) / y_test.shape[0]

print('Price up or down predict: %.2f'%(corr_price_development_test))
  
  
  
  
  
  
  
  
  
  
