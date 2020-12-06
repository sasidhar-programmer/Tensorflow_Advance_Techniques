import tensorflow as tf 

# calcuate (d/dw) w-square = 2w <-- formula  

x = tf.Variable([1.0])  

with tf.GradientTape() as tape : 
    grad = x * x 


tape.gradient(grad, x) 

