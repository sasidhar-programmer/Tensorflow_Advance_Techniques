import tensorflow as tf 
import numpy as np

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]) 
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])  

#trainable variables 
w = tf.Variable(np.random.random(), trainable=True) 
b = tf.Variable(np.random.random(), trainable=True) 


# loss 
def simple_loss(real_y, pred_y) : 
  return tf.abs(real_y - pred_y) 

learning_rate = 0.001

def fit_data(x_real, y_real): 

  with tf.GradientTape(persistent=True) as tape: 
    pred_y = w * x_real + b  
    reg_loss = simple_loss(y_real, pred_y) 

  w_gradient = tape.gradient(reg_loss, w) 
  b_gradient = tape.gradient(reg_loss, b) 

  w.assign_sub(w_gradient * learning_rate) 
  b.assign_sub(b_gradient * learning_rate) 


for _ in range(500) : 
  fit_data(xs, ys)  

print(f"w = {w.numpy()} + b = {b.numpy()}") 