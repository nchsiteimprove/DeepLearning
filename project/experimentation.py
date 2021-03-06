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

def test_acc():
    x_sym = T.imatrix()
    y_sym = T.imatrix()
    xmask_sym = T.matrix()

    output = np.zeros((0,2)).astype('int32')
    y = np.zeros((3,1)).astype('int32')
    print(y.shape)
    print(y.flatten().shape)

    output[0] = np.array([1, 2])
    output[1] = np.array([2, 3])
    output[2] = np.array([3, 4])

    y[0] = np.array([1])
    y[1] = np.array([0])
    y[2] = np.array([0])


    argmax0 = np.argmax(x_sym, axis=0)
    # print(argmax0)

    argmax1 = np.argmax(x_sym, axis=1)
    # print(argmax1)

    argmax_minus1 = np.argmax(x_sym, axis=-1)
    # print(argmax_minus1)

    argmax = argmax_minus1

    eq = T.eq(argmax, y_sym)
    # print(eq)
    acc = T.mean(eq)
    # print(acc)

    # func = theano.function([x_sym, y_sym], [eq, acc, argmax])
    func = theano.function([x_sym], [argmax])

    # eq_val, acc_val, argmax_val = func(output, y)
    argmax_val = func(output)

    # print("Eq:")
    # print(eq_val)
    # print("Acc:")
    # print(acc_val)
    print("Argmax:")
    print(argmax_val)

def test_embedding():
    VOCABULARY = 10

    x_sym = T.imatrix()
    x_data = range(VOCABULARY/2)
    X = np.zeros(shape=(1, len(x_data))).astype('int32')
    # X[0] = x_data
    X[0][0] = 1

    l_in = InputLayer(shape=(None, None))
    l_emb = EmbeddingLayer(l_in, VOCABULARY, VOCABULARY, W=np.eye(VOCABULARY, dtype='float32'))

    print("X:")
    print(X.shape)
    print(X)
    print("X embedded:")
    print(get_output(l_emb, inputs={l_in:x_sym}).eval({x_sym:X}).shape)
    print(get_output(l_emb, inputs={l_in:x_sym}).eval({x_sym:X}))

test_embedding()
