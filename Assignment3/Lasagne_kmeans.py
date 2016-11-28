from __future__ import print_function

import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import matplotlib.pylab as plt
from   PIL import Image
import csv

from util import *
import matplotlib.pyplot as plt
plt.ion()

num = 100




##### loading data and labels ####
def load_image():
    folder = "D:/UofT2016/Fall2016/MachineLearning/Project/data/train/"
    filename = os.listdir(folder)[:num]
    num_figs = len(filename)
    all_pixels = np.zeros((num_figs, 16384))  # np.zeros((num_figs, 16384,3))
    for i in range(num_figs):
        im = Image.open(folder + filename[i]).convert('L')
        pixel_values = np.array(im.getdata())
        pixel_values = np.reshape(pixel_values, 16384)
        all_pixels[i] = pixel_values
    return all_pixels


def load_label():
    with open('D:/UofT2016/Fall2016/MachineLearning/Project/data/train.csv', 'rb') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    your_list = np.array(your_list)
    ylabel = your_list[1:, 1].astype(int)[:num]
    # ylabel    = np.array(ylabel, dtype=float32)
    return ylabel

#################################  k-means  ############################


def distmat(p, q):
  """Computes pair-wise L2-distance between columns of p and q."""
  d, pn = p.shape
  d, qn = q.shape
  pmag = np.sum(p**2, axis=0).reshape(1, -1)
  qmag = np.sum(q**2, axis=0).reshape(1, -1)
  dist = qmag + pmag.T - 2 * np.dot(p.T, q)
  dist = (dist >= 0) * dist  # Avoid small negatives due to numerical errors.
  return np.sqrt(dist)

def KMeans(x, K, iters):
  """Cluster x into K clusters using K-Means.
  Inputs:
    x: Data matrix, with one data vector per column.
    K: Number of clusters.
    iters: Number of iterations of K-Means to run.
  Outputs:
    means: Cluster centers, with one cluster center in each column.
  """

  print("just entered kmeans")
  x
  N = x.shape[1]
  perm = np.arange(N)
  np.random.shuffle(perm)
  means = x[:, perm[:K]]
  dist = np.zeros((K, N))
  for ii in xrange(iters):
    print('Kmeans iteration = %04d' % (ii+1))
    for k in xrange(K):
        dist[k, :] = distmat(means[:, k].reshape(-1, 1), x)

    assigned_class = np.argmin(dist, axis=0)
    for k in xrange(K):
        means[:, k] = np.mean(x[:, (assigned_class == k).nonzero()[0]], axis=1)


  return means

def ShowMeans(means, number=0):
  """Show the cluster centers as images."""
  plt.figure(number)
  plt.clf()
  for i in xrange(means.shape[1]):
    plt.subplot(1, means.shape[1], i+1)
    plt.imshow(means[:, i].reshape(128, 128), cmap=plt.cm.gray)
  plt.draw()
  raw_input('Press Enter.')


############################## MOG #################################


def mogEM(x, K, iters, randConst=1, minVary=0):
    """
    Fits a Mixture of K Diagonal Gaussians on x.

    Inputs:
      x: data with one data vector in each column.
      K: Number of Gaussians.
      iters: Number of EM iterations.
      randConst: scalar to control the initial mixing coefficients
      minVary: minimum variance of each Gaussian.

    Returns:
      p: probabilities of clusters (or mixing coefficients).
      mu: mean of the clusters, one in each column.
      vary: variances for the cth cluster, one in each column.
      logLikelihood: log-likelihood of data after every iteration.
    """

    print ("just entered MOG")

    N, T = x.shape

    print("in MOG x shape is", x.shape)
    # Initialize the parameters
    p = randConst + np.random.rand(K, 1)
    p = p / np.sum(p)   # mixing coefficients
    mn = np.mean(x, axis=1).reshape(-1, 1)
    vr = np.var(x, axis=1).reshape(-1, 1)
    #mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)
    K = 8
    iter_kmeans = 5
    mu = KMeans(x, K, iter_kmeans)
    print("in mog mu shaoe is ", mu.shape)


    #------------------------------------------------------------
    vary = vr * np.ones((1, K)) * 2
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary

    logLikelihood = np.zeros((iters, 1))

    # Do iters iterations of EM
    for i in xrange(iters):
        # Do the E step
        respTot = np.zeros((K, 1))
        respX = np.zeros((N, K))
        respDist = np.zeros((N, K))
        ivary = 1 / vary
        logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - \
            0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)

        logPcAndx = np.zeros((K, T))
        for k in xrange(K):
            dis = (x - mu[:, k].reshape(-1, 1))**2
            logPcAndx[k, :] = logNorm[k] - 0.5 * \
                np.sum(ivary[:, k].reshape(-1, 1) * dis, axis=0)

        mx = np.max(logPcAndx, axis=0).reshape(1, -1)
        PcAndx = np.exp(logPcAndx - mx)
        Px = np.sum(PcAndx, axis=0).reshape(1, -1)
        PcGivenx = PcAndx / Px
        logLikelihood[i] = np.sum(np.log(Px) + mx)

        print ("Iter %d logLikelihood %.5f" , (i + 1, logLikelihood[i]))

        # Plot log likelihood of data
        plt.figure(0)
        plt.clf()
        plt.plot(np.arange(i), logLikelihood[:i], 'r-')
        plt.title('Log-likelihood of data versus # iterations of EM')
        plt.xlabel('Iterations of EM')
        plt.ylabel('Log-likelihood')
        plt.draw()

        # Do the M step
        # update mixing coefficients
        respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
        p = respTot

        # update mean
        respX = np.zeros((N, K))
        for k in xrange(K):
            respX[:, k] = np.mean(x * PcGivenx[k, :].reshape(1, -1), axis=1)

        mu = respX / respTot.T

        # update variance
        respDist = np.zeros((N, K))


        for k in xrange(K):
            respDist[:, k] = np.mean(
                (x - mu[:, k].reshape(-1, 1))**2 * PcGivenx[k, :].reshape(1, -1), axis=1)

        vary = respDist / respTot.T
        vary = (vary >= minVary) * vary + (vary < minVary) * minVary

    return p, mu, vary, logLikelihood


def mogLogLikelihood(p, mu, vary, x):
    """ Computes log-likelihood of data under the specified MoG model

    Inputs:
      x: data with one data vector in each column.
      p: probabilities of clusters.
      mu: mean of the clusters, one in each column.
      vary: variances for the cth cluster, one in each column.

    Returns:
      logLikelihood: log-likelihood of data after every iteration.
    """
    K = p.shape[0]
    N, T = x.shape
    ivary = 1 / vary
    logLikelihood = np.zeros(T)

    print(" mu shape is", mu.shape)
    for t in xrange(T):
        # Compute log P(c)p(x|c) and then log p(x)
        logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
            - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
            - 0.5 * \
            np.sum(ivary * (x[:, t].reshape(-1, 1) - mu)
                   ** 2, axis=0).reshape(-1, 1)

        mx = np.max(logPcAndx, axis=0)
        logLikelihood[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx

    return logLikelihood





def q2():
    # Question 4.2 and 4.3
    K = 7
    iters = 10
    minVary = 0.01
    randConst = 10

    # load data
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData(
        '../toronto_face.npz')

    # Train a MoG model with 7 components on all training data, i.e., inputs_train,
    # with both original initialization and kmeans initialization.
    #------------------- Add your code here ---------------------

    # original
    pii, u, var, logLikelihood =mogEM(inputs_train,K, iters, randConst, minVary)
    ShowMeans(u,1)
    ShowMeans(var, 2)
    plt.ion()
    plt.figure(3)
    plt.xlabel("Clusters")
    plt.ylabel("Pi")
    plt.bar(np.arange(len(pii)),pii)


"""


def lasagne_kmeans_mog():

    counter=1
    iters = 10
    minVary = 0.01
    randConst = 1.0

    numComponents = np.array([8])
    T = numComponents.shape[0]

    errorTrain = np.zeros(T)

    #errorTest = np.zeros(T)
    #errorValidation = np.zeros(T)

    x_train = load_image()
    print("here after loading images")
    y_train = load_label()
    print("here after loading labels")
    #print("labels are ", y_train)

    # images of each of the eight classes in the training set
    x_train_class1=  x_train[y_train==1]
    x_train_class2 = x_train[y_train==2]
    x_train_class3 = x_train[y_train==3]
    x_train_class4 = x_train[y_train==4]
    x_train_class5 = x_train[y_train==5]
    x_train_class6 = x_train[y_train==6]
    x_train_class7 = x_train[y_train==7]
    x_train_class8 = x_train[y_train==8]

    # number of images in each class in the training set
    num_class1_train = x_train_class1.shape[0]
    num_class2_train = x_train_class2.shape[0]
    num_class3_train = x_train_class3.shape[0]
    num_class4_train = x_train_class4.shape[0]
    num_class5_train = x_train_class5.shape[0]
    num_class6_train = x_train_class6.shape[0]
    num_class7_train = x_train_class7.shape[0]
    num_class8_train = x_train_class8.shape[0]

    log_likelihood_class = np.log(
        [num_class1_train, num_class2_train, num_class3_train, num_class4_train, num_class5_train,
        num_class6_train, num_class7_train, num_class8_train]) - np.log(num_class1_train + num_class2_train +
        num_class3_train + num_class4_train + num_class5_train + num_class6_train + num_class7_train + num_class8_train)



    for t in xrange(T):
        K = numComponents[t]

        # Train a MoG model with K components
        # Hints: using (x_train_anger, x_train_happy) train 2 MoGs

        p_1, mu_1, vary_1, log_likelihood_train_1 = mogEM(
            x_train_class1.T, K, iters, randConst=randConst, minVary=minVary)

        p_2, mu_2, vary_2, log_likelihood_train_2 = mogEM(
            x_train_class2.T, K, iters, randConst=randConst, minVary=minVary)

        p_3, mu_3, vary_3, log_likelihood_train_3 = mogEM(
            x_train_class3.T, K, iters, randConst=randConst, minVary=minVary)

        p_4, mu_4, vary_4, log_likelihood_train_4 = mogEM(
            x_train_class4.T, K, iters, randConst=randConst, minVary=minVary)

        p_5, mu_5, vary_5, log_likelihood_train_5 = mogEM(
            x_train_class5.T, K, iters, randConst=randConst, minVary=minVary)

        p_6, mu_6, vary_6, log_likelihood_train_6 = mogEM(
            x_train_class6.T, K, iters, randConst=randConst, minVary=minVary)

        p_7, mu_7, vary_7, log_likelihood_train_7 = mogEM(
            x_train_class7.T, K, iters, randConst=randConst, minVary=minVary)

       # p_8, mu_8, vary_8, log_likelihood_train_8 = mogEM(
        #    x_train_class8.T, K, iters, randConst=randConst, minVary=minVary)

    # Compute the probability P(d|x), classify examples, and compute error rate
        # Hints: using (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
        # to compute error rates, you may want to use mogLogLikelihood function

    log_likelihood_train_1 = mogLogLikelihood(
        p_1, mu_1, vary_1, x_train.T) + log_likelihood_class[0]

    log_likelihood_train_2 = mogLogLikelihood(
        p_2, mu_2, vary_2, x_train.T) + log_likelihood_class[1]

    log_likelihood_train_3 = mogLogLikelihood(
        p_3, mu_3, vary_3, x_train.T) + log_likelihood_class[2]

    log_likelihood_train_4 = mogLogLikelihood(
        p_4, mu_4, vary_4, x_train.T) + log_likelihood_class[3]

    log_likelihood_train_5 = mogLogLikelihood(
        p_5, mu_5, vary_5, x_train.T) + log_likelihood_class[4]

    log_likelihood_train_6 = mogLogLikelihood(
        p_6, mu_6, vary_6, x_train.T) + log_likelihood_class[5]

    log_likelihood_train_7 = mogLogLikelihood(
        p_7, mu_7, vary_7, x_train.T) + log_likelihood_class[6]

    #log_likelihood_train_8 = mogLogLikelihood(
     #   p_8, mu_8, vary_8, x_train.T) + log_likelihood_class[7]



       # predict_label_train = (log_likelihood_train_a <
                 #              log_likelihood_train_h).astype(float)



       # errorTrain[t] = np.sum(
        #    (predict_label_train != y_train).astype(float)) / y_train.shape[0]

        #errorValidation[t] = np.sum(
         #   (predict_label_valid != y_valid).astype(float)) / y_valid.shape[0]

       # errorTest[t] = np.sum(
        #    (predict_label_test != y_test).astype(float)) / y_test.shape[0]

   # print ("Training error rate = %.5f" , errorTrain[t])
    #print ("Validation error rate = %.5f" , errorValidation[t])
    #print ("Testing error rate = %.5f" , errorTest[t])

    # Plot the error rate
    plt.figure(0)
    plt.clf()
    #-------------------- Add your code here -------------------------------
    #------------------- Answers ---------------------
    # to be removed before release
    plt.plot(numComponents, errorTrain, 'r', label='Training')
    plt.plot(numComponents, errorValidation, 'g', label='Validation')
    plt.plot(numComponents, errorTest, 'b', label='Testing')
    plt.xlabel('Number of Mixture Components')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.draw()
    plt.pause(0.0001)

    """












if __name__ == '__main__':
    #-------------------------------------------------------------------------
    #lasagne_kmeans_mog()
    q2()
    raw_input('Press Enter to continue.')







"""
######################  main ##################################
def main():
  K = 7
  iters = 10
  inputs_train=load_image()
  print("here after loading images")
  target_train=load_label()
  print("here after loading labels")
  print("labels are ", target_train)

  means = KMeans(inputs_train, K, iters)
  print("after computing means")
  #ShowMeans(means, 0)
  print("means shape is ", means.shape)
  print("means are ", means)

  x_train_class1 = np.where(target_train == 1)[0]
  x_train_class2 = np.where(target_train == 2)[0]
  x_train_class3 = np.where(target_train == 3)[0]
  x_train_class4 = np.where(target_train == 4)[0]
  x_train_class5 = np.where(target_train == 5)[0]
  x_train_class6 = np.where(target_train == 6)[0]
  x_train_class7 = np.where(target_train == 7)[0]
  x_train_class8 = np.where(target_train == 8)[0]

  num_class1_train = x_train_class1.shape[1]
  num_class2_train = x_train_class2.shape[1]
  num_class3_train = x_train_class3.shape[1]
  num_class4_train = x_train_class4.shape[1]
  num_class5_train = x_train_class5.shape[1]
  num_class6_train = x_train_class6.shape[1]
  num_class7_train = x_train_class7.shape[1]
  num_class8_train = x_train_class8.shape[1]

  log_likelihood_class = np.log(
      [num_class1_train, num_class2_train, num_class3_train, num_class4_train, num_class5_train,
       num_class6_train, num_class7_train, num_class8_train]) -np.log(num_class1_train + num_class2_train +
       num_class3_train+ num_class4_train+num_class5_train+ num_class6_train+ num_class7_train+ num_class8_train)




  errorTrain = np.zeros(T)
  #errorTest = np.zeros(T)
  #errorValidation = np.zeros(T)

  log_likelihood_class = np.log(
      [num_anger_train, num_happy_train]) - np.log(num_anger_train + num_happy_train)



if __name__ == '__main__':
  main()

"""

