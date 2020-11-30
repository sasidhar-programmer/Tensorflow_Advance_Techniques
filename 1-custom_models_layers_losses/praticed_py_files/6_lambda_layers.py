import tensorflow as tf 
from tensorflow.keras import backend as K 



model = tf.keras.models.Sequential([ 

    tf.keras.layers.Flatten() ,
    tf.keras.layers.Dense(128) ,
    tf.keras.layers.Lambda(lambda x : tf.abs(x)) , 
                 # it takes values from previous layer and take absolute values of x 


    tf.keras.layers.Dense(10, activation= 'softmax')
])



#custom relu  function 
def my_relu(x):
    return K.maximum(-0.1, x)  


model = tf.keras.models.Sequential([ 
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(128) ,
    tf.keras.layers.Lambda(my_relu) , 
    tf.keras.layers.Dense(10, activation='softmax')

])



