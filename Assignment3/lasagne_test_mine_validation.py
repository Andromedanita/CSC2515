from __future__ import print_function

import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import matplotlib.pylab as plt
from   PIL              import Image
import csv

#num = int(sys.argv[1])

##### loading data and labels ####
def load_image(start_label, end_label):
    folder = "/Users/anita/Documents/Grad_Second_Year/CSC2515/assignment3/411a3/train/"
    filename = os.listdir(folder)[start_label:end_label]
    num_figs = len(filename)
    all_pixels = np.zeros((num_figs,3,128,128))  #np.zeros((num_figs, 16384,3))
    for i in range(num_figs):
        im  = Image.open(folder+filename[i])#.convert('L')
        pixel_values  = np.array(im.getdata())
        pixel_values  = np.reshape(pixel_values,(3,128,128))
        all_pixels[i] = pixel_values
    return all_pixels


def load_label(start_label, end_label):

    with open('/Users/anita/Documents/Grad_Second_Year/CSC2515/assignment3/411a3/train.csv','rb') as f:
        reader     = csv.reader(f)
        your_list  = list(reader)

    your_list = np.array(your_list)
    ylabel    = your_list[1:,1].astype(int)[start_label:end_label]
    return ylabel




def build_cnn(input_var=None):
    # We'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 3, 128, 128),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.
    
    
    #weights = np.ones((32,3,5,5))
    #weights = np.random.normal(loc=0.0, scale=1.0, size=2400)
    #weights = np.reshape(weights,(32,3,5,5))

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    
    
    
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    #W=weights)
            
   

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
            lasagne.layers.dropout(network, p=.1),
            num_units=8,
            nonlinearity=lasagne.nonlinearities.softmax)
            
    #print ("hvcsghvgefkghi",lasagne.layers.get_all_param_values(network))

    return network


# ############################# Batch iterator ###############################
'''
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



'''
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    
    class0   = np.where(targets == 0)[0]
    class1   = np.where(targets == 1)[0]
    class2   = np.where(targets == 2)[0]
    class3   = np.where(targets == 3)[0]
    class4   = np.where(targets == 4)[0]
    class5   = np.where(targets == 5)[0]
    class6   = np.where(targets == 6)[0]
    class7   = np.where(targets == 7)[0]
    #class8   = np.where(targets == 8)[0]
    
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        
        '''
        train_tmp = np.concatenate((class1[np.random.randint(0, len(class1), batchsize/8)],
                                    class2[np.random.randint(0, len(class2), batchsize/8)],
                                    class3[np.random.randint(0, len(class3), batchsize/8)],
                                    class4[np.random.randint(0, len(class4), batchsize/8)],
                                    class5[np.random.randint(0, len(class5), batchsize/8)],
                                    class6[np.random.randint(0, len(class6), batchsize/8)],
                                    class7[np.random.randint(0, len(class7), batchsize/8)],
                                    class8[np.random.randint(0, len(class8), batchsize/8)]))
        '''
        
        train_tmp = np.concatenate((class0[np.random.randint(0, len(class0), batchsize/8)],
                                    class1[np.random.randint(0, len(class1), batchsize/8)],
                                    class2[np.random.randint(0, len(class2), batchsize/8)],
                                    class3[np.random.randint(0, len(class3), batchsize/8)],
                                    class4[np.random.randint(0, len(class4), batchsize/8)],
                                    class5[np.random.randint(0, len(class5), batchsize/8)],
                                    class6[np.random.randint(0, len(class6), batchsize/8)],
                                    class7[np.random.randint(0, len(class7), batchsize/8)]))

        
        #print (":7777777", class7[np.random.randint(0, len(class7), batchsize/8)])
        #print (":8888888", class8[np.random.randint(0, len(class8), batchsize/8)])
                                    
        if shuffle:
            indices = train_tmp
            np.random.shuffle(indices)

        #print  ("expected target values: ", targets[train_tmp])
        yield inputs[train_tmp] , targets[train_tmp]

# ############################## Main program ################################

def main(num_epochs=20):
    # Load the dataset
    print("Loading data...")
    #X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    X_train, y_train = load_image(2900,3400), load_label(2900,3400)
    X_val, y_val = load_image(0,200), load_label(0,200)
    
    y_train -= 1
    y_val   -= 1

    # Prepare Theano variables for inputs and targets
    input_var  = T.tensor4('inputs')
    target_var = T.lvector('targets')


    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_var)
 
    #return

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    
    
    ##### prediction function ##########
    #predict_function = theano.function([input_var], prediction)
    predict_function = theano.function([input_var], T.argmax(prediction, axis=1))
    
    #accuracy_val = lasagne.objectives.categorical_accuracy(prediction, target_var, top_k=1)
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params  = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.momentum(
            loss, params, learning_rate=0.001, momentum=0.9)

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
    
    train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                                        dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        print ("Epoch is:", epoch)
        train_err     = 0
        train_batches = 0


        start_time    = time.time()
        for batch in iterate_minibatches(X_train, y_train, 160, shuffle=True):
            inputs, targets = batch
            train_err      += train_fn(inputs, targets)
            train_batches  += 1
        
        print ("train error is:",train_err)
        #print ("shape of prediction val = ", np.shape(pval))
        


        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 80, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        
        pval = predict_function(inputs)
        print ("shape of prediction val = ", np.shape(pval))
        print ("Prediction val = ", pval)

        # printing the results
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))


    print ("network params: ", lasagne.layers.get_all_params(network))
    print ("network values: ", lasagne.layers.get_all_param_values(network))
    print ("mean weights: ", np.mean(lasagne.layers.get_all_param_values(network)[0]))

    '''
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    '''
    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)
    #plt.xlim(0.5,100)

if __name__ == '__main__':
    plt.ion()
    main()


