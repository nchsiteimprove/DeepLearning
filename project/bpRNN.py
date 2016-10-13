%matplotlib inline
%matplotlib nbagg
import lasagne
import theano
import theano.tensor as T
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from lasagne.layers import InputLayer, GRULayer, DenseLayer, EmbeddingLayer, get_output


# Load data
data_filtered = load_training_data(5)
data_blocks_encoded, encodings = convert_training_data_individual_blocks(data_filtered, encode=True, statistics=True)

# Variables
BATCH_SIZE = 2
NUM_OUTPUTS = 2
VOCABULARY = len(encodings)
NUM_UNITS = 10
# Symbolic Theano variables
x_sym = T.imatrix()
y_sym = T.fmatrix()

# Define network layers
# Embedding layers
l_in = InputLayer(shape=(None, None))
l_emb = EmbeddingLayer(l_in, VOCABULARY, VOCABULARY, W=np.eye(VOCABULARY, dtype='float32'))

#TODO: Figure out exactly what this does
l_emb.params[l_emb.W].remove('trainable')

# Use dummy data to verify the shape of the embedding layer


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
