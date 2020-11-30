import tensorflow as tf 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D 


mnist = tf.keras.datasets.mnist   

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  

train_images = train_images / 255.0 
test_images = test_images / 255.0  



def model() : 

    input_layer = Input(shape = (28, 28))    

     
    flaten        = Flatten()(input_layer) 
    dense_1       = Dense(128, activation='relu')(flaten) 
    dense_2       = Dense(54, activation='relu')(dense_1) 
    output_layer  = Dense(10, activation='softmax')(dense_2)   

    model = Model(inputs = input_layer, outputs = output_layer) 
    return model 



model = model() 

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) 


model.fit(train_images, train_labels, epochs = 5) 

