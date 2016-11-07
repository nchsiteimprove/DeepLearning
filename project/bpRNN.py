import traceback
import time
import lasagne
import theano
import theano.tensor as T
import matplotlib
# from IPython import display
import numpy as np
import matplotlib.pyplot as plt
from lasagne.nonlinearities import linear, tanh
from lasagne.layers import InputLayer, GRULayer, DenseLayer, EmbeddingLayer, get_output, ReshapeLayer, SliceLayer, ConcatLayer, DropoutLayer
# from decoder_attention import LSTMAttentionDecodeFeedbackLayer
from training_data import get_batch, reset_batches, reset_train_batches, encodings, max_encoding, slice_list, get_nr_samples_to_process

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

def test_network(X, Y, Xmask, slice_size):
    accs, outputs, true_poss, true_negs, false_poss, false_negs, positives = [], [], [], [], [], [], []
    X_slices = slice_list(X, slice_size)
    Y_slices = slice_list(Y, slice_size)
    Xmask_slices = slice_list(Xmask, slice_size)
    for X_slice, Y_slice, Xmask_slice in zip(X_slices, Y_slices, Xmask_slices):
        acc_slice, output_slice, true_pos_slice, true_neg_slice, false_pos_slice, false_neg_slice, positives_slice = test_func(X_slice, Y_slice, Xmask_slice)

        accs += [acc_slice]
        outputs += [output_slice]
        true_poss += [true_pos_slice]
        true_negs +=  [true_neg_slice]
        false_poss += [false_pos_slice]
        false_negs += [false_neg_slice]
        positives += [positives_slice]

    acc = np.mean(accs)
    output = np.concatenate(outputs)
    true_pos = np.sum(true_poss)
    true_neg = np.sum(true_negs)
    false_pos = np.sum(false_poss)
    false_neg = np.sum(false_negs)
    positive = np.sum(positives)

    return acc, output, true_pos, true_neg, false_pos, false_neg, positive

def timing_human_readable(elapsed):
    if elapsed < 60:
        return elapsed, 's'

    minutes = elapsed / 60.0
    if minutes < 60:
        return minutes, 'm'

    hours = minutes / 60.0
    if hours < 24:
        return hours, 'h'

    days = hours / 24.0
    return days, 'd'

def calc_recall(true_pos, false_neg):
    divisor = true_pos + false_neg
    if divisor == 0:
        divisor = 1
    return float(true_pos) / divisor

def calc_precision(true_pos, false_pos):
    divisor = true_pos + false_pos
    if divisor == 0:
        divisor = 1
    return float(true_pos) / divisor

def calc_f1(precision, recall):
    divisor = precision+recall
    if divisor == 0:
        divisor = 1
    return 2 * float(precision*recall) / divisor

# Variables
NUM_OUTPUTS = 2
VOCABULARY = max_encoding + 1#len(encodings)
NUM_UNITS_ENC = 100 #TODO: Larger networks? Play around with hyper-parameters
NUM_UNITS_DEC = NUM_UNITS_ENC
NUM_UNITS_HID = NUM_UNITS_ENC
MAX_OUT_LABELS = 1
# Symbolic Theano variables
x_sym = T.imatrix()
y_sym = T.imatrix()
xmask_sym = T.matrix()

print("\nVocabulary size: %d"%VOCABULARY)

# Define network layers
# Embedding layers
l_in = InputLayer(shape=(None, None))
l_emb = EmbeddingLayer(l_in, VOCABULARY, VOCABULARY, W=np.eye(VOCABULARY, dtype='float32'))

l_emb.params[l_emb.W].remove('trainable')

# Use dummy data to verify the shape of the embedding layer
X, Y, Xmask = get_batch(3)

print("\nX: %s"%str(X.shape))
# allow_input_downcast risks loss of data
# print(get_output(l_emb, inputs={l_in:x_sym}).eval({x_sym:X}).shape)

###### Start of Encoder ######
l_mask_enc = InputLayer(shape=(None, None))

l_gru_enc_frwrd = GRULayer(incoming=l_emb, num_units=NUM_UNITS_ENC, mask_input=l_mask_enc)
# print(get_output(l_gru_enc_frwrd, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

# l_enc_frwrd_dropout = DropoutLayer(incoming=l_gru_enc_frwrd)

# l_gru_enc_frwrd2 = GRULayer(incoming=l_enc_frwrd_dropout, num_units=NUM_UNITS_ENC, mask_input=l_mask_enc)
# print(get_output(l_gru_enc_frwrd2, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)
#
# l_enc_frwrd_dropout2 = DropoutLayer(incoming=l_gru_enc_frwrd2)

l_gru_enc_bckwrd = GRULayer(incoming=l_emb, num_units=NUM_UNITS_ENC, mask_input=l_mask_enc, backwards=True)
# print(get_output(l_gru_enc_bckwrd, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

# l_enc_bcwrd_dropout = DropoutLayer(incoming=l_gru_enc_bckwrd)

# l_gru_enc_bckwrd2 = GRULayer(incoming=l_enc_bcwrd_dropout, num_units=NUM_UNITS_ENC, mask_input=l_mask_enc, backwards=True)
# print(get_output(l_gru_enc_bckwrd2, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)
#
# l_enc_bcwrd_dropout2 = DropoutLayer(incoming=l_gru_enc_bckwrd2)

l_enc = ConcatLayer([l_gru_enc_frwrd, l_gru_enc_bckwrd], axis=-1)
# print(get_output(l_enc, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

l_slice = SliceLayer(l_enc, indices=-1, axis=1)
# print(get_output(l_slice, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

###### End of Encoder ######

###### Start of Decoder ######
# l_in_rep = RepeatLayer(l_slice, n=MAX_OUT_LABELS)
# print("Repeat layer")
# print lasagne.layers.get_output(l_in_rep, inputs={l_in: x_sym, l_mask_enc: xmask_sym}).eval(
#     {x_sym: X, xmask_sym: Xmask}).shape
#
# l_gru_dec = GRULayer(incoming=l_in_rep, num_units=NUM_UNITS_DEC)
# print(get_output(l_gru_dec, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

# l_dec = LSTMAttentionDecodeFeedbackLayer(incoming=l_enc, num_units=NUM_UNITS_DEC)

# l_reshape = lasagne.layers.ReshapeLayer(l_enc, (-1, [2]))
# print(get_output(l_reshape, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

l_hid = DenseLayer(incoming=l_slice, num_units=NUM_UNITS_HID, nonlinearity=tanh)
# l_hid2 = DenseLayer(incoming=l_hid, num_units=NUM_UNITS_HID, nonlinearity=tanh)
# print(get_output(l_hid, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

### TODO: Maybe Exchange softmax for a sigmoid layer, as that is sufficient to describe our two classes?
l_softmax = DenseLayer(incoming=l_hid, num_units=NUM_OUTPUTS, nonlinearity=T.nnet.softmax)
# print(get_output(l_softmax, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

l_out = l_softmax
# l_out = lasagne.layers.ReshapeLayer(l_softmax, (x_sym.shape[0], -1, NUM_OUTPUTS))
# print(get_output(l_out, inputs={l_in:x_sym, l_mask_enc:xmask_sym}).eval({x_sym:X, xmask_sym:Xmask}).shape)

print("")
###### End of Decoder ######

# Define evaluation functions
output_train = get_output(l_out, inputs={l_in: x_sym, l_mask_enc:xmask_sym}, deterministic=False)
output_test = get_output(l_out, inputs={l_in: x_sym, l_mask_enc:xmask_sym}, deterministic=True)

# Cost function
reshaped = T.reshape(output_train, (-1, NUM_OUTPUTS))
flattened = y_sym.flatten()
# print("Reshaped: %s"%str(reshaped.shape))
# print("Flattened: %s"%str(flattened.shape))
### Binary cross entropy is designed to work on a scalar value and works well
### with a Sigmoid output for binary classification.
### Categorical cross entropy is designed to work on a vector of data
### and makes sense to use with a softmax output.
total_cost = T.nnet.categorical_crossentropy(reshaped, flattened)
mean_cost = T.mean(total_cost)

# Accuracy function
y_pred = T.argmax(output_train, axis=-1, keepdims=True)
eq = T.eq(y_pred, y_sym)
acc = T.mean(eq)

target = 1

c_true_pos = (T.eq(y_sym, target) * T.eq(y_pred, target)).sum()
c_true_neg = (T.neq(y_sym, target) * T.neq(y_pred, target)).sum()
c_false_pos = (T.neq(y_sym, target) * T.eq(y_pred, target)).sum()
c_false_neg = (T.eq(y_sym, target) * T.neq(y_pred, target)).sum()

c_total_examples = c_true_pos + c_true_neg + c_false_pos + c_false_neg
c_positives = y_sym.sum()

c_recall = c_true_pos / (c_true_pos + c_false_neg + 0.0001)
c_precision = c_true_pos / (c_true_pos + c_false_pos + 0.0001)
c_f1 = 2 * (c_precision * c_recall) / (c_precision + c_recall + 0.0001)

f1_cost = (1 - c_f1)

cost = f1_cost + (mean_cost / 10000)

all_parameters = lasagne.layers.get_all_params([l_out], trainable=True)

# print "Trainable Model Parameters"
# print "-"*40
# for param in all_parameters:
#     print param, param.get_value().shape
# print "-"*40

all_grads = [T.clip(g,-3,3) for g in T.grad(cost, all_parameters)]
all_grads = lasagne.updates.total_norm_constraint(all_grads,3)
learning_rate=0.01
updates = lasagne.updates.adam(all_grads, all_parameters, learning_rate=learning_rate)

train_func = theano.function([x_sym, y_sym, xmask_sym], [cost, acc, output_train, y_pred, eq, total_cost], updates=updates)

test_func = theano.function([x_sym, y_sym, xmask_sym], [acc, output_test, c_true_pos, c_true_neg, c_false_pos, c_false_neg, c_positives])

reset_batches()
# Generate validation data
slice_size = 50
Xval, Yval, Xmask_val = get_batch(50)
Xtest, Ytest, Xmask_test = get_batch(150, store_train_index=True)
# print "Xval", Xval.shape
# print "Yval", Yval.shape

# TRAINING
BATCH_SIZE = 40
val_interval = BATCH_SIZE*5
samples_to_process = get_nr_samples_to_process()
nr_epochs = 200

reduce_learning_rate_n_epochs = 10
reduce_learning_rate_factor = 1.5

samples_processed = 0
last_valid_samples = 0

print("Training...")
output_folder = "output/"
val_samples, train_samples = [], []
costs, accs_val, accs_train = [], [], []
batch_durations = []
# plt.figure()
verbose = True
debug = False

converge_after = 20
converge_batch = -1
converge_epoch = -1
converge_value = -1
converge_steps = 0

print_timing = False
try:
    for i_epoch in range(nr_epochs):
        samples_processed = 0
        last_valid_samples = 0
        batch_count = 0
        reset_train_batches()

        if i_epoch != 0 and i_epoch % reduce_learning_rate_n_epochs == 0:
            if verbose:
                print("Decreasing learning rate")
            learning_rate /= reduce_learning_rate_factor

        if verbose:
            print("Epoch %d"%i_epoch)

        while samples_processed < samples_to_process:
            t_start = time.time()
            if verbose:
                print("Batch %d"%batch_count)
                batch_count += 1
            if debug:
                print("\tGetting batch")
            x_, ys_, x_masks_ = \
                get_batch(BATCH_SIZE)
            # if len(ids_) == 1:
            #     print(ids_[0])
            # else:
            #     print(len(ids_))
            if x_ is None or len(x_) == 0:
                break
            if debug:
                print("\tTraining batch")
            batch_cost, batch_acc, batch_output, batch_y_pred, batch_eq, batch_total_cost = train_func(x_, ys_, x_masks_)

            ### Converge check start
            converge_on = batch_acc
            if converge_on != converge_value:
                converge_epoch = i_epoch
                converge_batch = batch_count
                converge_value = converge_on
                converge_steps = 0

            else:
                converge_steps += 1

            if converge_steps >= converge_after:
                if verbose:
                    print("\t\t\t\tNetwork has converged!")

            ### Converge check end

            if verbose:
                print("\tTrain accuracy: %.2f%%"%(batch_acc * 100))
            if debug:
                print("Output:")
                print(batch_output)
                # print("Labels:")
                # print(ys_)
                print("Cost:")
                print(batch_total_cost)
                # print("Output shape:")
                # print(batch_output.shape)
                # print("Argmax shape:")
                # print(batch_y_pred.shape)
                # print("Eq shape:")
                # print(batch_eq.shape)

            samples_processed += BATCH_SIZE

            #validation data
            # if verbose:
            #     print("\tPossible validation")
            #TODO: Collect samples and plot only when validating
            if samples_processed - last_valid_samples > val_interval:
                print_timing = True
                last_valid_samples = samples_processed

                ## Only track these here to get more readable graphs
                costs += [batch_cost]
                accs_train += [batch_acc]
                train_samples += [samples_processed]

                #print "validating"
                if verbose:
                    print("\tValidating network")
                # val_accs, val_outputs = [], []
                # Xval_slices = slice_list(Xval, slice_size)
                # Yval_slices = slice_list(Yval, slice_size)
                # Xmask_val_slices = slice_list(Xmask_val, slice_size)
                # for Xval_slice, Yval_slice, Xmask_val_slice in zip(Xval_slices, Yval_slices, Xmask_val_slices):
                #     val_acc_slice, val_output_slice = test_func(Xval_slice, Yval_slice, Xmask_val_slice)
                #     val_accs += [val_acc_slice]
                #     val_outputs += [val_output_slice]
                # val_acc = np.mean(val_accs)
                # val_output = np.mean(val_outputs)

                val_acc, val_output, true_pos, true_neg, false_pos, false_neg, positive = test_network(Xval, Yval, Xmask_val, slice_size)

                recall = calc_recall(true_pos, false_neg)
                precision = calc_precision(true_pos, false_pos)
                f1 = calc_f1(precision, recall)

                # print(val_output)
                if verbose:
                    # print("\tTrain Accuracy: %.2f%%"%(batch_acc*100))
                    print("\tValid Accuracy: %.2f%%"%(val_acc*100))
                    print("\tValid F1: %.2f%%"%(f1*100))
                val_samples += [samples_processed]
                accs_val += [val_acc]

                ## Make acc png
                # plt.clf()
                # plt.plot(val_samples,accs_val, label='validation')
                # plt.title('', fontsize=20)
                # plt.grid('on')
                # plt.plot(train_samples,accs_train, label='train')
                # plt.ylabel('Accuracy', fontsize=15)
                # plt.xlabel('Processed samples', fontsize=15)
                # plt.title('', fontsize=20)
                # plt.grid('on')
                # plt.legend(loc='best')
                # plt.savefig(output_folder + "acc_.png")

                # plt.plot(val_samples,accs_val)
                # plt.ylabel('Validation Accuracy', fontsize=15)
                # plt.xlabel('Processed samples', fontsize=15)
                # plt.title('', fontsize=20)
                # plt.grid('on')
                # plt.savefig(output_folder + "acc_val.png")
                # display.display(display.Image(filename="out.png"))
                # display.clear_output(wait=True)
            t_end = time.time()
            t_dur = t_end - t_start
            batch_durations.append(t_dur)

            if verbose and print_timing:
                print_timing = False
                t_spent = sum(batch_durations)
                t_batch_avg = t_spent / float(len(batch_durations))

                batches_left = ((samples_to_process - samples_processed) / BATCH_SIZE) * (nr_epochs - i_epoch)
                t_left = batches_left * t_batch_avg

                t, form = timing_human_readable(t_spent)
                print("\tTime spent: %.2f%s"%(t, form))
                t, form = timing_human_readable(t_left)
                print("\tEstimated time left: %.2f%s"%(t, form))

except KeyboardInterrupt:
    pass
except:
    traceback.print_exc()


## Make valid acc png
# plt.plot(val_samples,accs_val, label='valid acc')
# plt.ylabel('Validation Accuracy', fontsize=15)
# plt.xlabel('Processed samples', fontsize=15)
# plt.title('', fontsize=20)
# plt.grid('on')
# plt.savefig(output_folder + "acc_val.png")

# print(len(costs))
# print((costs[0]))
# Save the legend to the acc png
# plt.legend(loc='best')
# plt.savefig(output_folder + "acc.png")
# Make train cost png
# plt.clf()
# plt.plot(train_samples,costs, label='cost')
# plt.ylabel('Train costs', fontsize=15)
# plt.xlabel('Processed samples', fontsize=15)
# plt.title('', fontsize=20)
# plt.grid('on')
# plt.legend(loc='best')
# plt.savefig(output_folder + "cost_train_.png")

# print(len(accs_train))
# print((accs_train[0]))
## Make train acc png
# plt.plot(train_samples,accs_train, label='train acc')
# plt.ylabel('Train Accuracy', fontsize=15)
# plt.xlabel('Processed samples', fontsize=15)
# plt.title('', fontsize=20)
# plt.grid('on')
# plt.savefig(output_folder + "acc_train.png")

if converge_steps >= converge_after:
    print("\nNetwork converged at epoch %d, batch %d"%(converge_epoch, converge_batch))

print("\nTraining done, calculating final result...")
t_start = time.time()
test_acc, test_output, true_pos, true_neg, false_pos, false_neg, positive = test_network(Xtest, Ytest, Xmask_test, slice_size)

# nr_examples = 0
# nr_correct = 0
# guesses_acc = []
# for y_out, y_label in zip(test_output, Ytest):
#     nr_examples += 1
#     guess = np.argmax(y_out)
#     if debug:
#         print(y_out)
#         print(guess)
#     if guess == y_label:
#         nr_correct += 1
#         guesses_acc.append(1)
#     else:
#         guesses_acc.append(0)
#
# verified_acc = nr_correct / float(nr_examples) * 100.0
# verified_acc_mean = np.mean(guesses_acc) * 100.0
#
# np_max = np.argmax(test_output, axis=-1)
# np_test = np.equal(np_max, Ytest)
# np_ones_zeros = []
# for b in np_test:
#     if b[0]:
#         np_ones_zeros.append(1)
#     else:
#         np_ones_zeros.append(0)
# verified_acc_mean_np = np.mean(np_ones_zeros) * 100.0

t_end = time.time()
t_dur = t_end - t_start

batch_durations.append(t_dur)
t_total = sum(batch_durations)
t, form = timing_human_readable(t_dur)
print("Test time: %.2f%s"%(t, form))
t, form = timing_human_readable(t_total)
print("Total time: %.2f%s"%(t, form))

print("Accuracy: %.2f%%"%(test_acc*100))
# print("Verified accuracy: %.2f%%"%verified_acc)
# print("Verified accuracy mean: %.2f%%"%verified_acc_mean)
# print("Verified accuracy mean numpy: %.2f%%"%verified_acc_mean_np)

print("True positives: %d"%true_pos)
print("False positives: %d"%false_pos)
print("True negatives: %d"%true_neg)
print("False negatives: %d"%false_neg)
print("of %d training examples"%len(test_output))
print("with %d positive labels"%positive)

recall = calc_recall(true_pos, false_neg)
precision = calc_precision(true_pos, false_pos)
f1 = calc_f1(precision, recall)
print("")
print("Precision: %.2f%%"%(precision * 100.0))
print("Recall: %.2f%%"%(recall * 100.0))
print("F1: %.2f%%"%(f1 * 100.0))
