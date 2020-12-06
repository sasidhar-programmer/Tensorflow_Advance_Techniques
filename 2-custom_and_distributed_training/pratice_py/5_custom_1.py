import tensorflow as tf 
from tensorflow.keras.layers import Dense, Flatten, Input 
from tensorflow.keras.models import Model 


def base_model() : 

    inputs = Input(shape = (), name = 'clothing')  
    x = Dense(64, activation= 'relu', name = 'dense_1')(inputs) 
    x = Dense(64, activation= 'relu', name = 'dense_2')(x) 
    output = Dense(10, activation= 'softmax', name = 'output') 

    model = Model(inputs = inputs, outputs = output) 
    return model 


