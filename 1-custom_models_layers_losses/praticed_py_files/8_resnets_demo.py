import tensorflow as tf 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Layer 
from tensorflow.keras.models import Model 


class CNNResidual(Model) :

    def __init__(self, layers, filters, **kwargs) : 
        super(CNNResidual, self).__init__(**kwargs)  

        self.HiddenCNN = [Conv2D(filters, (3,3), activation='relu') for _ in range(layers)] 



    def call(self, inputs) : 
        x = inputs 
        for layer in self.HiddenCNN: 
            x = layer(x) 
            return inputs + x 


class DNNResidual(Model) : 

    def __init__(self, layers, neurons, **kwargs): 
        super(DNNResidual, self).__init__(**kwargs)  

        self.HiddenDNN = [Dense(neurons, activation='relu') for _ in range(layers)]  


    def call(self, inputs) : 
        x = inputs 
        for layer in self.HiddenDNN : 
            x = layer(x) 
        return inputs + x  


class MyResidual(Model) : 

    def __init__(self, **kwargs) : 
        super(MyResidual, self).__init__(self) 

        self.hidden1 = Dense(60, activation= 'relu') 
        self.block1 = CNNResidual(3, 64) 
        self.block2 = DNNResidual(3, 64)   
        self.out = Dense(1) 


    def call(self, inputs) :  

        x = self.hidden1(inputs) 
        x = self.block1(x) 
        for _ in range(1, 4): 
            x = self.block2(x) 

        return self.out(x) 
