import tensorflow as tf 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Layer ,Add ,BatchNormalization, Activation

from tensorflow.keras.models import Model   

class IdentityBlock(Model) : 
    def __init__(self, filters, kernal_size,  **kwargs) : 
        super(IdentityBlock, self).__init__(**kwargs)  

        self.conv1 = Conv2D(filters, kernal_size, padding='same') 
        self.bn1 = BatchNormalization() 

        self.act = Activation('relu') 

        self.add = Add() 


    def call(self, inputs) : 

        x = self.conv1(inputs) 
        x = self.bn1(x) 
        x = self.act(x)  

        x = self.conv1(x) 
        x = self.bn1(x) 
        x = self.act(x) 


        x = self.add([x, inputs]) 
        x = self.act(x) 

        return x 



class ResNet(Model) : 
    
    def __init__(self, num_classes, **kwargs) : 
        super(ResNet, self).__init__(**kwargs)  

        self.conv = Conv2D(64, 7, padding= 'same') 
        self.bn = BatchNormalization() 
        self.act = Activation('relu') 

        self.max_pool = tf.keras.layers.MaxPool2D((3,3)) 

        self.idblock1 = IdentityBlock(64, 3) 
        self.idblock2 = IdentityBlock(64, 3) 

        self.glob_pool = tf.keras.layers.GlobalAveragePooling2D() 
        
        self.classifier = Dense(num_classes, activation= 'softmax') 


    def call(self, inputs) : 

        x = self.conv(inputs) 
        x = self.bn(x) 
        x = self.act(x) 

        x = self.max_pool(x) 

        x = self.idblock1(x) 
        x = self.idblock2(x) 

        x = self.glob_pool(x) 

        x = self.classifier(x) 

        return x 


resnet = ResNet(10) 


