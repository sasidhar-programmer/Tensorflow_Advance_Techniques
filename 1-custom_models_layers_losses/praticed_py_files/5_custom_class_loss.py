import tensorflow as tf 
from tensorflow import keras  
from tensorflow.keras.losses import Loss 

# custom loss with hyperparameter 

def my_huber_loss_with_threshold(threshold): 
    
    def my_huber_loss(y_true, y_pred) : 
        error =  y_true - y_pred 
        is_small_error = tf.abs(error) <= threshold 
        small_error_loss = tf.square(error) / 2
        big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
        
        return tf.where(is_small_error, small_error_loss, big_error_loss) 

    # return the inner function tuned by the hyperparameter
    return my_huber_loss 

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss=my_huber_loss_with_threshold(threshold=1.2)) #calling loss 



# CUSTOM LOSS AS A CLASS 


class MyHuberLoss(Loss) : 
    
    threshold = 1 

    def __init__(self, threshold): 
        super().__init__() 
        self.threshold = threshold 

    def call(self, y_true, y_pred) : 
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= self.threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = self.threshold * (tf.abs(error) - (0.5 * self.threshold))
        return tf.where(is_small_error, small_error_loss, big_error_loss)



model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss=MyHuberLoss(threshold=1.02)) #calling loss 


