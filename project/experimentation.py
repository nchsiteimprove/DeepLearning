import traceback
import lasagne
import theano
import theano.tensor as T
import matplotlib
# from IPython import display
import numpy as np
import matplotlib.pyplot as plt
from lasagne.layers import InputLayer, GRULayer, DenseLayer, EmbeddingLayer, get_output, ReshapeLayer, SliceLayer
# from training_data import get_batch, reset_batches, encodings, max_encoding

x_sym = T.imatrix()
y_sym = T.imatrix()
xmask_sym = T.matrix()

output = np.zeros((3,2)).astype('int32')
y = np.zeros((3,2)).astype('int32')

output[0] = np.array([1, 0])
output[1] = np.array([0, 1])
output[2] = np.array([0, 1])

y[0] = np.array([1, 0])
y[1] = np.array([0, 1])
y[2] = np.array([0, 1])


argmax0 = np.argmax(x_sym, axis=0)
# print(argmax0)

argmax1 = np.argmax(x_sym, axis=1)
# print(argmax1)

argmax_minus1 = np.argmax(x_sym, axis=-1)
# print(argmax_minus1)

argmax = argmax0

eq = T.eq(argmax, y_sym)
# print(eq)
acc = T.mean(eq)
# print(acc)

func = theano.function([x_sym, y_sym], [eq, acc, argmax])

eq_val, acc_val, argmax_val = func(output, y)

print(eq_val)
print(acc_val)
print(argmax_val)
