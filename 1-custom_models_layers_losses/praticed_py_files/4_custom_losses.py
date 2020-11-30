# DON'T RUN THIS CODE 

import  tensorflow as tf 
from tensorflow import keras  

# code for custom huber loss 

def my_huber_loss(y_true, y_pred): 

    threshold = 1
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= threshold
    small_error_loss = tf.square(error) / 2
    big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
    return tf.where(is_small_error, small_error_loss, big_error_loss) 


model = tf.keras.models.Sequential([ tf.keras.layers.Dense(20, activation='relu')])

model.compile(optimizer = 'sgd', loss = my_huber_loss) # using custom loss 

