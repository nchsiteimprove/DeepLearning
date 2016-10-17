
import lasagne
import theano
import theano.tensor as T
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from lasagne.layers import InputLayer, GRULayer, DenseLayer, EmbeddingLayer, get_output
from training_data import get_batch, reset_batches, encodings

# Variables
BATCH_SIZE = 2
NUM_OUTPUTS = 2
VOCABULARY = len(encodings)
NUM_UNITS = 10
# Symbolic Theano variables
x_sym = T.fmatrix()
y_sym = T.fmatrix()
xmask_sym = T.matrix()

# Define network layers
# Embedding layers
l_in = InputLayer(shape=(None, None))
l_emb = EmbeddingLayer(l_in, VOCABULARY, VOCABULARY, W=np.eye(VOCABULARY, dtype='float32'))

#TODO: Figure out exactly what this does
l_emb.params[l_emb.W].remove('trainable')

# Use dummy data to verify the shape of the embedding layer
X, y, Xmask = get_batch(2)

print(X.shape)
# allow_input_downcast risks loss of data
result = get_output(l_emb, inputs={l_in:x_sym}).eval({x_sym:X})
rshape = result.shape
print(rshape)

# Encoder layers

l_gru = GRULayer(incoming=l_in, num_units=NUM_UNITS, only_return_final=True)
l_out = DenseLayer(incoming=l_gru, num_units=NUM_OUTPUTS, nonlinearity=T.nnet.softmax)

# Define evaluation functions
output_train = get_output(l_out, inputs={l_in: x_sym}, deterministic=False)

total_cost = T.nnet.categorical_crossentropy(T.reshape(output_train, (-1, NUM_OUTPUTS)), y_sym.flatten())
mean_cost = T.mean(total_cost)

all_parameters = lasagne.layers.get_all_params([l_out], trainable=True)

updates = lasagne.updates.adam(all_grads, all_parameters, learning_rate=0.005)

f_train = theano.function([x_sym, y_sym], [cost, output_train], updates=updates)
