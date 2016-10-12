%matplotlib inline
%matplotlib nbagg
import lasagne
import theano
import theano.tensor as T
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


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

#TODO: Design a way to load our data instead
batch_size = 3
inputs, input_masks, targets, target_masks, text_inputs, text_targets = \
    get_batch(batch_size=batch_size,max_digits=2,min_digits=1)

print "input types:", inputs.dtype,  input_masks.dtype, targets.dtype, target_masks.dtype
print print_valid_characters()
print "Stop character = #"

for i in range(batch_size):
    print "\nSAMPLE",i
    print "TEXT INPUTS:\t\t", text_inputs[i]
    print "TEXT TARGETS:\t\t", text_targets[i]
    print "ENCODED INPUTS:\t\t", inputs[i]
    print "MASK INPUTS:\t\t", input_masks[i]
    print "ENCODED TARGETS:\t", targets[i]
    print "MASK TARGETS:\t\t", target_masks[i]

#######################################
BATCH_SIZE = 100
NUM_UNITS_ENC = 10
NUM_UNITS_DEC = 10
#TODO: Change names of these vars to something more fitting
MAX_DIGITS = 20
MIN_DIGITS = MAX_DIGITS #currently only support for same length outputs - we'll leave it for an exercise to add support for varying length targets
NUM_INPUTS = 27
NUM_OUTPUTS = 11 #(0-9 + '#')


#symbolic theano variables. Note that we are using imatrix for X since it goes into the embedding layer
x_sym = T.imatrix()
y_sym = T.imatrix()
xmask_sym = T.matrix()

#dummy data to test implementation - We advise to check the output-dimensions of all layers.
#One way to do this in lasagne/theano is to forward pass some data through the model and
#check the output dimensions of these.
#Create some random testdata
#TODO: Create some random test data fitting our problem
X = np.random.randint(0,10,size=(BATCH_SIZE,MIN_DIGITS)).astype('int32')
Xmask = np.ones((BATCH_SIZE,MIN_DIGITS)).astype('float32')

##### ENCODER START #####
l_in = lasagne.layers.InputLayer((None, None))
l_emb = lasagne.layers.EmbeddingLayer(l_in, NUM_INPUTS, NUM_INPUTS,
                                      W=np.eye(NUM_INPUTS,dtype='float32'),
                                      name='Embedding')
#Here we'll remove the trainable parameters from the embeding layer to constrain
#it to a simple "one-hot-encoding". You can experiment with removing this line
l_emb.params[l_emb.W].remove('trainable')
#forward pass some data throug the inputlayer-embedding layer and print the output shape
print lasagne.layers.get_output(l_emb, inputs={l_in: x_sym}).eval({x_sym: X}).shape

l_mask_enc = lasagne.layers.InputLayer((None, None))
l_enc = lasagne.layers.GRULayer(l_emb, num_units=NUM_UNITS_ENC, name='GRUEncoder', mask_input=l_mask_enc)
print lasagne.layers.get_output(l_enc, inputs={l_in: x_sym, l_mask_enc: xmask_sym}).eval(
    {x_sym: X, xmask_sym: Xmask}).shape

# slice last index of dimension 1
l_last_hid = lasagne.layers.SliceLayer(l_enc, indices=-1, axis=1)
print lasagne.layers.get_output(l_last_hid, inputs={l_in: x_sym, l_mask_enc: xmask_sym}).eval(
    {x_sym: X, xmask_sym: Xmask}).shape
##### END OF ENCODER######


##### START OF DECODER######
l_in_rep = RepeatLayer(l_last_hid, n=MAX_DIGITS+1) #we add one to allow space for the end of sequence character
print lasagne.layers.get_output(l_in_rep, inputs={l_in: x_sym, l_mask_enc: xmask_sym}).eval(
    {x_sym: X, xmask_sym: Xmask}).shape

l_dec = lasagne.layers.GRULayer(l_in_rep, num_units=NUM_UNITS_DEC, name='GRUDecoder')
print lasagne.layers.get_output(l_dec, inputs={l_in: x_sym, l_mask_enc: xmask_sym}).eval(
    {x_sym: X, xmask_sym: Xmask}).shape


# We need to do some reshape voodo to connect a softmax layer to the decoder.
# See http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html#examples
# In short this line changes the shape from
# (batch_size, decode_len, num_dec_units) -> (batch_size*decodelen,num_dec_units).
# We need to do this since the softmax is applied to the last dimension and we want to
# softmax the output at each position individually
l_reshape = lasagne.layers.ReshapeLayer(l_dec, (-1, [2]))
print lasagne.layers.get_output(l_reshape, inputs={l_in: x_sym, l_mask_enc: xmask_sym}).eval(
    {x_sym: X, xmask_sym: Xmask}).shape

l_softmax = lasagne.layers.DenseLayer(l_reshape, num_units=NUM_OUTPUTS,
                                      nonlinearity=lasagne.nonlinearities.softmax,
                                      name='SoftmaxOutput')
print lasagne.layers.get_output(l_softmax, inputs={l_in: x_sym, l_mask_enc: xmask_sym}).eval(
    {x_sym: X, xmask_sym: Xmask}).shape

# reshape back to 3d format (batch_size, decode_len, num_dec_units). Here we tied the batch size to the shape of the symbolic variable for X allowing
#us to use different batch sizes in the model.
l_out = lasagne.layers.ReshapeLayer(l_softmax, (x_sym.shape[0], -1, NUM_OUTPUTS))
print lasagne.layers.get_output(l_out, inputs={l_in: x_sym, l_mask_enc: xmask_sym}).eval(
    {x_sym: X, xmask_sym: Xmask}).shape
###END OF DECODER######

output_decoder_train = lasagne.layers.get_output(l_out, inputs={l_in: x_sym, l_mask_enc: xmask_sym},
                                                deterministic=False)

#cost function
total_cost = T.nnet.categorical_crossentropy(
    T.reshape(output_decoder_train, (-1, NUM_OUTPUTS)), y_sym.flatten())
mean_cost = T.mean(total_cost)
#accuracy function
argmax = T.argmax(output_decoder_train,axis=-1)
eq = T.eq(argmax,y_sym)
acc = T.mean(eq)  # gives float64 because eq is uint8, T.cast(eq, 'float32') will fix that...

#Get parameters of both encoder and decoder
all_parameters = lasagne.layers.get_all_params([l_out], trainable=True)

print "Trainable Model Parameters"
print "-"*40
for param in all_parameters:
    print param, param.get_value().shape
print "-"*40

#add grad clipping to avoid exploding gradients
all_grads = [T.clip(g,-3,3) for g in T.grad(mean_cost, all_parameters)]
all_grads = lasagne.updates.total_norm_constraint(all_grads,3)

#Compile Theano functions.
updates = lasagne.updates.adam(all_grads, all_parameters, learning_rate=0.005)
train_func = theano.function([x_sym, y_sym, xmask_sym], [mean_cost, acc, output_decoder_train], updates=updates)
#since we don't have any stochasticity in the network we will just use the training graph without any updates given
test_func = theano.function([x_sym, y_sym, xmask_sym], [acc, output_decoder_train])


#Generate validation data
#TODO: Change this to load our data
Xval, Xmask_val, Yval, Ymask_val, text_inputs_val, text_targets_val = \
    get_batch(batch_size=5000, max_digits=MAX_DIGITS,min_digits=MIN_DIGITS)
print "Xval", Xval.shape
print "Yval", Yval.shape

### TRAINING ###
val_interval = 5000
samples_to_process = 3e5
samples_processed = 0

val_samples = []
costs, accs = [], []
plt.figure()
try:
    while samples_processed < samples_to_process:
        #TODO: Change to get batch of our data
        x_, x_masks_, ys_, y_masks_, _, _ = \
            get_batch(batch_size=BATCH_SIZE,max_digits=MAX_DIGITS,min_digits=MIN_DIGITS)
        batch_cost, batch_acc, batch_output = train_func(x_, ys_, x_masks_)
        costs += [batch_cost]
        samples_processed += BATCH_SIZE
        #validation data
        if samples_processed % val_interval == 0:
            #print "validating"
            val_acc, val_output = test_func(Xval, Yval, Xmask_val)
            val_samples += [samples_processed]
            accs += [val_acc]
            plt.plot(val_samples,accs)
            plt.ylabel('Validation Accuracy', fontsize=15)
            plt.xlabel('Processed samples', fontsize=15)
            plt.title('', fontsize=20)
            plt.grid('on')
            plt.savefig("out.png")
            display.display(display.Image(filename="out.png"))
            display.clear_output(wait=True)
except KeyboardInterrupt:
    pass

#plot of validation accuracy for each target position
plt.figure(figsize=(7,7))
plt.plot(np.mean(np.argmax(val_output,axis=2)==Yval,axis=0))
plt.ylabel('Accuracy', fontsize=15)
plt.xlabel('Target position', fontsize=15)
#plt.title('', fontsize=20)
plt.grid('on')
plt.show()


### Added to test the output
batch_size = 1
inputs, input_masks, targets, target_masks, text_inputs, text_targets = \
    get_batch(batch_size=batch_size,max_digits=MAX_DIGITS,min_digits=MIN_DIGITS)

# test_func = theano.function([x_sym, y_sym, xmask_sym], [acc, output_decoder_train])

# Xval, Xmask_val, Yval, Ymask_val, text_inputs_val, text_targets_val = \
#     get_batch(batch_size=5000, max_digits=MAX_DIGITS,min_digits=MIN_DIGITS)

# print "input types:", inputs.dtype,  input_masks.dtype, targets.dtype, target_masks.dtype
# print print_valid_characters()
# print "Stop character = #"

for i in range(batch_size):
#     print "\nSAMPLE",i
    print "TEXT INPUTS:\t\t", text_inputs[i]
    print "TEXT TARGETS:\t\t", text_targets[i]
#     print "ENCODED INPUTS:\t\t", inputs[i]
#     print "MASK INPUTS:\t\t", input_masks[i]
#     print "ENCODED TARGETS:\t", targets[i]
#     print "MASK TARGETS:\t\t", target_masks[i]

val_acc, val_output = test_func(inputs, targets, input_masks)

print("RESULTS:")

prediction = ""
for digit in val_output:
    for out in digit:
#         print(out)
        idx = np.argmax(out)
        if idx == NUM_OUTPUTS - 1:
            idx = '#'
        prediction += str(idx)

print(prediction)
