import tensorflow as tf 
from tensorflow.keras.layers import Layer 

class Variables(Layer) : 
    
    def __init__(self): 
        super().__init__() 
        self.one = tf.Variable(100)  
        self.two = [tf.Variable(x) for x in range(4)]  

    
var = Variables() 

print([varss.numpy() for varss in var.variables])
