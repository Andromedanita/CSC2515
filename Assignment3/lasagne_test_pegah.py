
from __future__ import print_function
import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from PIL import Image
import csv

num=5

def load_image():
    folder = "/Users/anita/Documents/Grad_Second_Year/CSC2515/assignment3/411a3/train/"
    filename = os.listdir(folder)[:num]
    num_figs = len(filename)
    all_pixels = np.zeros((num_figs, 3, 128, 128))  # np.zeros((num_figs, 16384,3))
    for i in range(num_figs):
        im = Image.open(folder + filename[i])  # .convert('L')
        pixel_values = np.array(im.getdata())
        pixel_values = np.reshape(pixel_values, (3, 128, 128))
        all_pixels[i] = pixel_values
    return all_pixels


def load_train_labels():
    with open('/Users/anita/Documents/Grad_Second_Year/CSC2515/assignment3/411a3/train.csv', 'rb') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    your_list = np.array(your_list)
    ylabel = your_list[1:, 1].astype(int)[:num]
    return ylabel


# We can now download and read the training and test set images and labels.
X_train = load_image()
y_train = load_train_labels()


def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    # none: unspecified batch size
    network = lasagne.layers.InputLayer(shape=(None, 3, 128, 128),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    #print("in iterate mini batch")
    #print("input length")
    #print(len(inputs))
    #print(batchsize)
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
            #print("in first if")
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
            #print("in else before yield")
        #print("before yield")
        yield inputs[excerpt], targets[excerpt]





def main(num_epochs=10):

    #print("i am herreeeee")

    # Load the dataset
    print("Loading data...")
    X_train= load_image()
    y_train=load_train_labels()

    # Prepare Theano variables for inputs and targets
    input_var  = T.tensor4('inputs')
    target_var = T.lvector('targets') #T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

    network = build_cnn(input_var)


    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        print ("Epoch is:", epoch)
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 1, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
            #print("batchhhh")

    print ("i am done")
    print("train error is ")
    print(train_err)


    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

main()