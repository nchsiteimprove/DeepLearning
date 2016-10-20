import traceback
import lasagne
import theano
import theano.tensor as T
import matplotlib
# from IPython import display
import numpy as np
import matplotlib.pyplot as plt
from lasagne.nonlinearities import linear
from lasagne.layers import InputLayer, GRULayer, DenseLayer, EmbeddingLayer, get_output, ReshapeLayer, SliceLayer, ConcatLayer
from training_data import get_batch, reset_batches, encodings, max_encoding

class RepeatLayer(lasagne.layers.Layer):
    def __init__(self, incoming, n, **kwargs):
        '''
        The input is expected to be a 2D tensor of shape
        (num_batch, num_features). The input is repeated
        n times such that the output will be
        (num_batch, n, num_features)
        '''
        super(RepeatLayer, self).__init__(incoming, **kwargs)
        self.n = n

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[0], self.n] + list(input_shape[1:]))

    def get_output_for(self, input, **kwargs):
        #repeat the input n times
        tensors = [input]*self.n
        stacked = theano.tensor.stack(*tensors)
        dim = [1, 0] + range(2, input.ndim + 1)
        return stacked.dimshuffle(dim)

# Variables
NUM_OUTPUTS = 2
VOCABULARY = max_encoding + 1#len(encodings)
NUM_UNITS_ENC = 10
NUM_UNITS_DEC = NUM_UNITS_ENC
MAX_OUT_LABELS = 1
# Symbolic Theano variables
x_sym = T.imatrix()
y_sym = T.imatrix()
xmask_sym = T.matrix()

print("Vocabulary size: %d"%VOCABULARY)

# Define network layers
# Embedding layers
l_in = InputLayer(shape=(None, None))
l_emb = EmbeddingLayer(l_in, VOCABULARY, VOCABULARY, W=np.eye(VOCABULARY, dtype='float32'))

l_emb.params[l_emb.W].remove('trainable')

# Use dummy data to verify the shape of the embedding layer
X, Y, Xmask = get_batch(3)

print("X: %s"%str(X.shape))
# allow_input_downcast risks loss of data
print(get_output(l_emb, inputs={l_in:x_sym}).eval({x_sym:X}).shape)

###### Start of Encoder ######
l_mask_enc = InputLayer(shape=(None, None))

l_gru_enc_frwrd = GRULayer(incoming=l_emb, num_units=NUM_UNITS_ENC, mask_input=l_mask_enc)
print(get_output(l_gru_enc_frwrd, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

l_gru_enc_bckwrd = GRULayer(incoming=l_emb, num_units=NUM_UNITS_ENC, mask_input=l_mask_enc, backwards=True)
print(get_output(l_gru_enc_bckwrd, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

l_gru_enc = ConcatLayer([l_gru_enc_frwrd, l_gru_enc_bckwrd], axis=-1)
print(get_output(l_gru_enc, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

l_slice = SliceLayer(l_gru_enc, indices=-1, axis=1)
print(get_output(l_slice, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

###### End of Encoder ######

###### Start of Decoder ######
l_in_rep = RepeatLayer(l_slice, n=MAX_OUT_LABELS)
print("Repeat layer")
print lasagne.layers.get_output(l_in_rep, inputs={l_in: x_sym, l_mask_enc: xmask_sym}).eval(
    {x_sym: X, xmask_sym: Xmask}).shape

l_gru_dec = GRULayer(incoming=l_in_rep, num_units=NUM_UNITS_DEC)
print(get_output(l_gru_dec, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

l_reshape = lasagne.layers.ReshapeLayer(l_gru_dec, (-1, [2]))
print(get_output(l_reshape, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

### TODO: Exchange softmax for a sigmoid layer, as that is sufficient to describe our two classes
l_softmax = DenseLayer(incoming=l_reshape, num_units=NUM_OUTPUTS, nonlinearity=T.nnet.softmax)
print(get_output(l_softmax, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

l_out = lasagne.layers.ReshapeLayer(l_softmax, (x_sym.shape[0], -1, NUM_OUTPUTS))
print(get_output(l_out, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

###### End of Decoder ######

# Define evaluation functions
output_train = get_output(l_out, inputs={l_in: x_sym, l_mask_enc:xmask_sym}, deterministic=False)
output_test = get_output(l_out, inputs={l_in: x_sym, l_mask_enc:xmask_sym}, deterministic=True)

# Cost function
reshaped = T.reshape(output_train, (-1, NUM_OUTPUTS))
flattened = y_sym.flatten()
# print("Reshaped: %s"%str(reshaped.shape))
# print("Flattened: %s"%str(flattened.shape))
total_cost = T.nnet.categorical_crossentropy(reshaped, flattened)
mean_cost = T.mean(total_cost)

# Accuracy function
argmax = T.argmax(output_train, axis=-1)#, keepdims=True)
eq = T.eq(argmax, y_sym)
acc = T.mean(eq)

all_parameters = lasagne.layers.get_all_params([l_out], trainable=True)

print "Trainable Model Parameters"
print "-"*40
for param in all_parameters:
    print param, param.get_value().shape
print "-"*40

all_grads = [T.clip(g,-3,3) for g in T.grad(mean_cost, all_parameters)]
all_grads = lasagne.updates.total_norm_constraint(all_grads,3)

updates = lasagne.updates.adam(all_grads, all_parameters, learning_rate=0.005)

train_func = theano.function([x_sym, y_sym, xmask_sym], [mean_cost, acc, output_train, argmax, eq, total_cost], updates=updates)

test_func = theano.function([x_sym, y_sym, xmask_sym], [acc, output_test])

reset_batches()
# Generate validation data
Xval, Yval, Xmask_val = get_batch(5)#1000)
# print "Xval", Xval.shape
# print "Yval", Yval.shape

# TRAINING
BATCH_SIZE = 2#200
val_interval = BATCH_SIZE*10
samples_to_process = 20#200000
samples_processed = 0
last_valid_samples = 0

print("Training...")
val_samples = []
costs, accs = [], []
plt.figure()
verbose = True
debug = True
c = 1
try:
    while samples_processed < samples_to_process:
        if verbose:
            print("Batch %d"%c)
            c += 1
            print("\tGetting batch")
        x_, ys_, x_masks_ = \
            get_batch(BATCH_SIZE)
        if x_ is None:
            break
        if verbose:
            print("\tTraining batch")
        batch_cost, batch_acc, batch_output, batch_argmax, batch_eq, batch_total_cost = train_func(x_, ys_, x_masks_)
        if debug:
            print("Output:")
            print(batch_output)
            print("Labels:")
            print(ys_)
            print("Cost:")
            print(batch_total_cost)
            # print("Output shape:")
            # print(batch_output.shape)
            # print("Argmax shape:")
            # print(batch_argmax.shape)
            # print("Eq shape:")
            # print(batch_eq.shape)
        costs += [batch_cost]
        samples_processed += BATCH_SIZE
        #validation data
        # if verbose:
        #     print("\tPossible validation")
        if samples_processed - last_valid_samples > val_interval:
            last_valid_samples = samples_processed
            #print "validating"
            if verbose:
                print("\tValidating network")
            val_acc, val_output = test_func(Xval, Yval, Xmask_val)
            # print(val_output)
            if verbose:
                print("\tAccuracy: %.2f%%"%val_acc)
            val_samples += [samples_processed]
            accs += [val_acc]
            plt.plot(val_samples,accs)
            plt.ylabel('Validation Accuracy', fontsize=15)
            plt.xlabel('Processed samples', fontsize=15)
            plt.title('', fontsize=20)
            plt.grid('on')
            plt.savefig("out.png")
            # display.display(display.Image(filename="out.png"))
            # display.clear_output(wait=True)
except KeyboardInterrupt:
    pass
except:
    traceback.print_exc()

print("Training done, final result")
val_acc, val_output = test_func(Xval, Yval, Xmask_val)
print("\tAccuracy: %.2f%%"%val_acc)
