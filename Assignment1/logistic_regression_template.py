from check_grad import check_grad
from utils      import *
from logistic   import *


def run_logistic_regression(hyperparameters):
    # TODO specify training data
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()

    # N is number of examples; M is the number of features per example.
    N, M = train_inputs.shape

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    #weights     = np.random.normal(0, 1./hyperparameters['alpha'], M+1)
    weights = np.random.rand(M+1)

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for t in xrange(hyperparameters['num_iterations']):

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)
        
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        print "iteration:", t
        print "first weight: ",np.shape(weights)
        # update parameters
        print "multiply", np.shape(hyperparameters['learning_rate'] * df/N )
        
        weights = weights - (hyperparameters['learning_rate'] * df / N)[:,0]

        #print "iteration:", t
        print "second weight: ", np.shape(weights)
        print "validation: ", np.shape(valid_inputs)
        # Make a prediction on the valid_inputs.
        print "stupiiiiiid code"
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)

        # print some stats
        print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
               "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                   t+1, f / N, cross_entropy_train, frac_correct_train*100,
                   cross_entropy_valid, frac_correct_valid*100)
        #print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}").format(t+1, f / N)
        #print "entropy: ", cross_entropy_valid
        #print "frac: ", frac_correct_valid
        
        logging[t] = [f / N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100]
    return logging

def run_check_grad(hyperparameters):
    """
    Performs gradient check on logistic function.
    """

    # This creates small random data with 7 examples and 
    # 9 dimensions and checks the gradient on that data.
    num_examples   = 7
    num_dimensions = 9

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = (np.random.rand(num_examples, 1) > 0.5).astype(int)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print "diff =", diff

if __name__ == '__main__':
    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.5,
                    'weight_regularization': False, # boolean, True for using Gaussian prior on weights
                    'num_iterations': 700,
                    'weight_decay': 1. # related to standard deviation of weight prior 
                    }

    # average over multiple runs
    num_runs = 1
    logging = np.zeros((hyperparameters['num_iterations'], 5))
    for i in xrange(num_runs):
        logging += run_logistic_regression(hyperparameters)
    logging /= num_runs

    # TODO generate plots
    #[...]
