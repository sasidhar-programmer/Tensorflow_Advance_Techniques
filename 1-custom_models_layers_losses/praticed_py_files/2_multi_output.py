# THIS IS SAMPLE CODE 
#      DON'T RUN IT 

import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Flatten, Dense , Input  

input_layer = Input( shape = (None, None)) 

layer_1 = Dense(64, activation= 'relu', name ='deses_1')(input_layer) 
layer_2 = Dense(32, activation= tf.nn.tanh(), name = 'dense_2')(layer_1) 


y_1_output = Dense(1, activation=tf.nn.sigmoid())(layer_2)


layer_3 = Dense( 128, activation= tf.nn.relu())(layer_2) 

y_2_output = Dense(1, activation=tf.nn.sigmoid())(layer_3)

model = Model(inputs = input_layer, outputs = [y_1_output, y_2_output])

# compile the model 

model.compile(optimizer = tf.keras.optimizers.Adam(), 
                loss = { 'y_1_output' : 'mse', 
                          'y_2_output': "mae" 
                        }, 
                metircs = { 'y_1_output' : tf.keras.metrics.RootMeanSquaredError() ,
                            'y_2_output' : tf.keras.metircs.RootMeanSquaredError() 
                         }
                )    




                





