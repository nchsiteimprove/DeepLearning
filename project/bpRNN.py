import traceback
import lasagne
import theano
import theano.tensor as T
import matplotlib
# from IPython import display
import numpy as np
import matplotlib.pyplot as plt
from lasagne.layers import InputLayer, GRULayer, DenseLayer, EmbeddingLayer, get_output, ReshapeLayer, SliceLayer
from training_data import get_batch, reset_batches, encodings, max_encoding

# Variables
NUM_OUTPUTS = 2
VOCABULARY = max_encoding#len(encodings)
NUM_UNITS = 10
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

# Encoder layers
l_mask_enc = InputLayer(shape=(None, None))
l_gru_enc = GRULayer(incoming=l_emb, num_units=NUM_UNITS, mask_input=l_mask_enc)
print(get_output(l_gru_enc, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

l_slice = SliceLayer(l_gru_enc, indices=-1, axis=1)
print(get_output(l_slice, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

# l_reshape = lasagne.layers.ReshapeLayer(l_slice, (-1, [2]))
# print(get_output(l_reshape, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

l_softmax = DenseLayer(incoming=l_slice, num_units=NUM_OUTPUTS, nonlinearity=T.nnet.softmax)
print(get_output(l_softmax, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

# l_out = lasagne.layers.ReshapeLayer(l_softmax, (x_sym.shape[0], -1, NUM_OUTPUTS))
# print(get_output(l_out, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

l_out = DenseLayer(incoming=l_softmax, num_units=NUM_OUTPUTS)
print(get_output(l_out, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

# Define evaluation functions
output_train = get_output(l_out, inputs={l_in: x_sym, l_mask_enc:xmask_sym}, deterministic=False)

# Cost function
reshaped = T.reshape(output_train, (-1, NUM_OUTPUTS))
flattened = y_sym.flatten()
# print("Reshaped: %s"%str(reshaped.shape))
# print("Flattened: %s"%str(flattened.shape))
total_cost = T.nnet.categorical_crossentropy(reshaped, flattened)
mean_cost = T.mean(total_cost)

# Accuracy function
argmax = T.argmax(output_train, axis=-1)
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

train_func = theano.function([x_sym, y_sym, xmask_sym], [mean_cost, acc, output_train], updates=updates)

test_func = theano.function([x_sym, y_sym, xmask_sym], [acc, output_train])

reset_batches()
# Generate validation data
Xval, Yval, Xmask_val = get_batch(3)
print "Xval", Xval.shape
print "Yval", Yval.shape

# TRAINING
BATCH_SIZE = 3
val_interval = 2
samples_to_process = 7
samples_processed = 0

print("Training...")
val_samples = []
costs, accs = [], []
plt.figure()
verbose = True
c = 1
try:
    while samples_processed < samples_to_process:
        if verbose:
            print("Batch %d"%c)
            c += 1
            print("\tGetting batch")
        x_, ys_, x_masks_ = \
            get_batch(BATCH_SIZE)
        if verbose:
            print("\tTraining batch")
        batch_cost, batch_acc, batch_output = train_func(x_, ys_, x_masks_)
        costs += [batch_cost]
        samples_processed += BATCH_SIZE
        #validation data
        if verbose:
            print("\tPossible validation")
        if samples_processed % val_interval == 0:
            #print "validating"
            if verbose:
                print("\tTesting network")
            val_acc, val_output = test_func(Xval, Yval, Xmask_val)
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
