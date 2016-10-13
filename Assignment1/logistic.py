""" Methods for doing logistic regression."""

import numpy as     np
from   utils import sigmoid


def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities of being second class. This is the output of the classifier.
    """

    '''
    # TODO: Finish this function
    yarray = np.zeros(np.shape(data)[0])
    
    for i in range(np.shape(data)[0]):
        val       = np.dot(weights[:-1].T,data[i]) + weights[-1] #last element is w0 (bias)
        yarray[i] = np.exp(-val)/(1+np.exp(-val)) # maybe sigmoid function- second class=2s or 8s

    return yarray
    '''
    # adding 1 to the end of each row (to each image as its 785th element)
    data = np.insert(data, np.shape(data)[1], 1, axis=1)

    # dot product of w.x including w0 because I added 1 to the data set
    z    = np.dot(weights.T, data.T)

    # sigmoid for P(C=1 | x,w)
    y    = np.exp(-z)/(1+np.exp(-z))

    # transposing it to make the shape correct
    y    = y.T
    
    return y 

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """

    # calculating corss-entropy
    ce = 0
    for i in range(len(targets)):
        ce += -(targets[i] * np.log(y[i])) - ((1-targets[i]) * np.log(1-y[i]))
    
    # making predictions smaller than 0.5 to be 0
    # and predictions larger than 0.5 to be 1
    boundary       = 0.5
    y              = np.array(y)
    y[y>=boundary] = 1.0
    y[y<boundary]  = 0.0

    y = np.array(y)
    # checking similarities between prediction and target
    num_correct = 0

    # counting the number of correct predictions with respect to targets
    for i in range(len(y)):
        if (targets[i][0] == y[i]):
            num_correct += 1

    # fraction of correct predictions
    frac_correct = float(num_correct)/len(y) 
    return ce[0], frac_correct # ce[0] is the correct shape that works with other functions


def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
        y:       N x 1 vector of probabilities.
    """

    # predictions from sigmoid function for P(C=1| x,w)
    y = logistic_predict(weights, data)

    # conditions for using logistic regression or regularization
    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
        
    else:
        # compute f and df without regularization

        f  = 0
        df = np.zeros(len(weights))
        for i in range(len(data)):
            f += -targets[i] * np.log(y[i]) - (1-targets[i]) * np.log(1-y[i]) # f values 

        #for j in range(len(weights)-1):
        for j in range(len(weights)-1): 
            weights_sum = 0 # initial value is 0
            # for every image (i)
            for i in range(len(data)):
                x = data[i][j]
                weights_sum += x * (targets[i] - y[i]) #sum of x_i * (t_i -y_i)

            df[j] = weights_sum

        # doing same thing for bias (last element)    
        bias_val = 0    
        for i in range(len(data)):
            bias_val    += targets[i] - y[i]
        df[len(weights)-1] = bias_val

    # making df in correct shape for the rest of the functions    
    df = np.array([df]).T

    return f[0], df, y # f[0] so that it returns one number instead of a list


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of accumulative derivative of f w.r.t. weights, i.e. don't need to average over number of sample
    """

    '''
    # TODO: Finish this function
    df = hyperparameters['lambda']* np.sum((y - target)*data)
    z  = weights[1:]*data + weights[0]
    f  = np.sum(target * z) + np.sum(np.log(1+np.exp(-z) ) )
                    
    
    return f, df
    '''
    return 0
