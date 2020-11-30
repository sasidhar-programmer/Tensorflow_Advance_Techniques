# don't run code 

import tensorflow as tf 
from tensorflow.keras.layers import Dense, Flatten, Lambda  , Input 
from tensorflow.keras.models import Model 
from tensorflow.keras import backend as K 
from tensorflow.keras.utils import plot_model 

def base_model() : 

    inputs = Input(shape = (28, 28), name = 'input_layer') 
    x = Flatten(name = 'Flatten_layer')(inputs) 
    x = Dense(64, activation= 'relu')(x) 
    x = Dense(32, activation= 'relu')(x) 

    return Model(inputs = inputs, outputs = x) 


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1) 


# initialize the base network   

base_network = base_model()  

# create the left input and point to the base network

input_a = Input(shape=(28,28,), name="left_input")
vect_output_a = base_network(input_a)

# create the right input and point to the base network

input_b = Input(shape=(28,28,), name="right_input")
vect_output_b = base_network(input_b)

# measure the similarity of the two vector outputs 
# lambda layer gives flexibility for custom layers

output = Lambda(euclidean_distance, name="output_layer", 
                output_shape=eucl_dist_output_shape)([vect_output_a, vect_output_b])

# specify the inputs and output of the model
model = Model([input_a, input_b], output) 

plot_model(model, to_file = 'siamese.png', show_shapes= True, show_layer_names=True) 


#loss for siamese network 
def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss