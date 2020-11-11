import tensorflow as tf
from tensorflow import nn
import numpy as np


# data io
# data I/O
n, p = 0, 0
seq_length = 25
learning_rate = 1e-1
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]



# model = tf.keras.Sequential
inputs = tf.keras.layers.Input(shape=(vocab_size,2))
conv1 = tf.keras.layers.Conv1D(vocab_size,2,strides=5,activation=nn.relu)(inputs)
conv2 = tf.keras.layers.Conv1D(vocab_size,2,strides=5,activation=nn.relu)(conv1)
dense1 = tf.keras.layers.Dense(vocab_size, activation=nn.relu)(conv2)
outputs = tf.keras.layers.Dense(vocab_size,activation=nn.softmax) (dense1) 

model = tf.keras.Model(inputs=inputs, outputs=outputs)
print(outputs[0][0][0:24])

model_loss=tf.nn.l2_loss(outputs[0][0][0:25],targets)


model.compile(optimizer='adam',loss=model_loss) 
model.summary()
# model.fit(x=inputs,y=targets)
# model.predict()


while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    # hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data


  if n % 100 == 0:
    sample_ix =  model.predict(inputs[0],steps=200) # sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print ('----\n %s \n----' % (txt, ))
   
  p += seq_length # move data pointer
  n += 1 # iteration counter  