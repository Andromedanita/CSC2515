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
    print("in mog mu shape is ", mu.shape)


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
        print("mx shape is ", mx.shape)
        print(mx)
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





def kmeans_mog():
    # Question 4.2 and 4.3
    print("here in q2")
    K = 8
    iters = 20
    minVary = 0.01
    randConst = 10

    # load data
    #inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData(
     #   '../toronto_face.npz')

    inputs_train=load_image()
    target_train=load_label()

    # Train a MoG model with 7 components on all training data, i.e., inputs_train,
    # with both original initialization and kmeans initialization.
    #------------------- Add your code here ---------------------

    # original
    pii, u, var, logLikelihood =mogEM(inputs_train.T,K, iters, randConst, minVary)
    #ShowMeans(u,1)
    #ShowMeans(var, 2)
    plt.ion()
    plt.figure(3)
    plt.xlabel("Clusters")
    plt.ylabel("Pi")
    plt.bar(np.arange(len(pii)),pii)





if __name__ == '__main__':
    #-------------------------------------------------------------------------
    print("here in main")
    kmeans_mog()
    raw_input('Press Enter to continue.')


