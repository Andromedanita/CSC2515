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
    #dot_vals = map(lambda x: np.dot(weights[:-1].T, x) + weights[-1], data)
    #y        = map(lambda x: np.exp(-x)/(1+np.exp(-x)), dot_vals)

    data = np.insert(data, np.shape(data)[1], 1, axis=1)
    z    = np.dot(weights.T, data.T)

    y    = np.exp(-z)/(1+np.exp(-z))
    y    = y.T
    print "y is", np.shape(y)
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
    
    ce = 0

    for i in range(len(targets)):
        ce += -(targets[i] * np.log(y[i])) - ((1-targets[i]) * np.log(1-y[i]))

    
    # makign predictions smaller than 0.5 to be 0
    # and predictions larger than 0.5 to be 1
    boundary = 0.5

    y = np.array(y)
    y[y>=boundary] = 1.0
    y[y<boundary]  = 0.0

    y = np.array(y)
    # checking similarities between prediction and target
    num_correct = 0


    for i in range(len(y)):
        if (targets[i][0] == y[i]):
            #print "i inside = ", i
            num_correct += 1
    #num_correct = np.sum(targets == y)

    frac_correct = float(num_correct)/len(y) 
    return ce[0], frac_correct


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

    y = logistic_predict(weights, data)
    print np.shape(y), len(y)
    
    
    if hyperparameters['weight_regularization'] is True:
        f, df = logistic_pen(weights, data, targets, hyperparameters)
        
    else:
        # TODO: compute f and df without regularization

        f = 0
        df   = np.zeros(len(weights))
        for i in range(len(data)):
            f += -targets[i] * np.log(y[i]) - (1-targets[i]) * np.log(1-y[i]) # f values 

        #for j in range(len(weights)-1):
        for j in range(len(weights)-1): 
            weights_sum = 0
            for i in range(len(data)):
                x = data[i][j]
                weights_sum += x * (targets[i] - y[i])

            df[j] = weights_sum

        bias_val = 0    
        for i in range(len(data)):
            bias_val    += targets[i] - y[i]
        df[len(weights)-1] = bias_val

        
    df = np.array([df]).T
    #print "f", f[0]
    #print type(f)
    return f[0], df, y


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
