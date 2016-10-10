"""
Anita Bahmanyar
student number: 998909098
"""

import numpy       as     np
from   l2_distance import l2_distance

def run_knn(k, train_data, train_labels, valid_data):
    """Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples, 
          and M is the number of features per example.

    Inputs:
        k:            The number of neighbours to use for classification 
                      of a validation example.
        train_data:   The N_TRAIN x M array of training
                      data.
        train_labels: The N_TRAIN x 1 vector of training labels
                      corresponding to the examples in train_data 
                      (must be binary).
        valid_data:   The N_VALID x M array of data to
                      predict classes for.

    Outputs:
        valid_labels: The N_VALID x 1 vector of predicted labels 
                      for the validation data.
    """

    # TODO call l2_distance to compute distance between valid data and train data
    dist      = l2_distance(train_data.T, valid_data.T)
    # TODO sort the distance to get top k nearest data
    num_valid = np.shape(valid_data)[0]
    knearest  = np.zeros((num_valid, k)) # has the shape of (50,5) for instance
    nearest   = np.zeros((num_valid, k))
    
    # storing k nearest values for each test image in an array
    for i in range(num_valid):
        knearest[i] = sorted(dist[:,i])[:k]

    nearest_indx = np.zeros((num_valid, k))    
    for i in range(num_valid):
        for j in range(k):
            nearest_indx[i][j] = np.where(sorted(dist[:,i])[:k][j] == dist[:,i])[0][0] 
            nearest[i][j]      = train_labels[nearest_indx[i][j]]

            
    train_labels = train_labels.reshape(-1)
    #valid_labels = train_labels[nearest]
    valid_labels = nearest  
    
    # note this only works for binary labels
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1,1)

    return valid_labels
