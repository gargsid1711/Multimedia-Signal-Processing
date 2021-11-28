import os, h5py
from PIL import Image
import numpy as  np

###############################################################################
# TODO: here are the functions that you need to write
def todo_dataset_mean(X):
    '''
    mu = todo_dataset_mean(X)
    Compute the average of the rows in X (you may use any numpy function)
    X (NTOKSxNDIMS) = data matrix
    mu (NDIMS) = mean vector
    '''
    X_shape = X.shape
    mu = np.zeros(X_shape[1])

    for i in range(len(mu)):
        mu[i] = np.average(X[:, i])

    return mu

def todo_center_datasets(train, dev, test, mu):
    '''
    ctrain, cdev, ctest = todo_center_datasets(train, dev, test, mu)
    Subtract mu from each row of each matrix, return the resulting three matrices.
    '''
    ctrain = np.zeros_like(train)
    cdev = np.zeros_like(dev)
    ctest = np.zeros_like(test)

    ctrain = train - mu
    cdev = dev - mu
    ctest = test - mu

    return ctrain, cdev, ctest

def todo_find_transform(X):
    '''
    V, Lambda = todo_find_transform(X)
    X (NTOKS x NDIM) - data matrix.  You may assume that NDIM > NTOKS
    V (NDIM x NTOKS) - The first NTOKS principal component vectors of X
    Lambda (NTOKS) - The  first NTOKS eigenvalues of the covariance or gram matrix of X

    Find and return the PCA transform for the given X matrix:
    a matrix in which each column is a principal component direction.
    You can assume that the # data is less than the # dimensions per vector,
    so you should probably use the gram-matrix method, not the covariance method.
    Standardization: Make sure that each of your returned vectors has unit norm,
    and that its first element is non-negative.
    Return: (V, Lambda)
      V[:,i] = the i'th principal component direction
      Lambda[i] = the variance explained by the i'th principal component

    V and Lambda should both be sorted so that Lambda is in descending order of absolute
    value.  Notice: np.linalg.eig doesn't always do this, and in fact, the order it generates
    is different on my laptop vs. the grader, leading to spurious errors.  Consider using
    np.argsort and np.take_along_axis to solve this problem, or else use np.linalg.svd instead.
    '''

    u, s, vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)
    vh = vh/np.linalg.norm(vh)

    return np.transpose(vh), np.square(s)

def todo_transform_datasets(ctrain, cdev, ctest, V):
    '''
    ttrain, tdev, ttest = todo_transform_datasets(ctrain, cdev, ctest, V)
    ctrain, cdev, ctest are each (NTOKS x NDIMS) matrices (with different numbers of tokens)
    V is an (NDIM x K) matrix, containing the first K principal component vectors

    Transform each x using transform, return the resulting three datasets.
    '''

    ttrain = np.dot(ctrain, V)
    tdev = np.dot(cdev, V)
    ttest = np.dot(ctest, V)

    return ttrain, tdev, ttest

def todo_distances(train,test,size):
    '''
    D = todo_distances(train, test, size)
    train (NTRAINxNDIM) - one training vector per row
    test (NTESTxNDIM) - one test vector per row
    size (scalar) - number of dimensions to be used in calculating distance
    D (NTRAIN x NTEST) - pairwise Euclidean distances between vectors

    Return a matrix D such that D[i,j]=distance(train[i,:size],test[j,:size])
    '''
    D = np.zeros((len(train), len(test)))

    for i in range(len(train)):
        for j in range(len(test)):
            D[i, j] = np.linalg.norm(train[i, :size] - test[j, :size])


    return D

def todo_nearest_neighbor(Ytrain, D):
    '''
    hyps = todo_nearest_neighbor(Ytrain, D)
    Ytrain (NTRAIN) - a vector listing the class indices of each token in the training set
    D (NTRAIN x NTEST) - a matrix of distances from train to test vectors
    hyps (NTEST) - a vector containing a predicted class label for each test token

    Given the dataset train, and the (NTRAINxNTEST) matrix D, returns
    an int numpy array of length NTEST, specifying the person number (y) of the training token
    that is closest to each of the NTEST test tokens.
    '''

    dims = D.shape
    hyps = np.zeros(dims[1])

    for i in range(len(hyps)):
        index = np.argmin(D[i])
        hyps[i] = Ytrain[index]

    return hyps

def todo_compute_accuracy(Ytest, hyps):
    '''
    ACCURACY, CONFUSION = todo_compute_accuracy(TEST, HYPS)
    TEST (NTEST) - true label indices of each test token
    HYPS (NTEST) - hypothesis label indices of each test token
    ACCURACY (scalar) - the total fraction of hyps that are correct.
    CONFUSION (4x4) - confusion[ref,hyp] is the number of class "ref" tokens (mis)labeled as "hyp"
    '''
    C = np.zeros((4, 4))

    for i in range(len(Ytest)):
        C[Ytest[i], int(hyps[i])] += 1

    correct = 0
    for i in range(len(C)):
        for j in range(len(C)):
            if i == j:
                correct += C[i, j]
    accuracy = correct/np.sum(C)

    return accuracy, C

def todo_find_bestsize(ttrain, tdev, Ytrain, Ydev, variances):
    '''
    BESTSIZE, ACCURACIES = todo_find_bestsize(TTRAIN, TDEV, YTRAIN, YDEV, VARIANCES)
    TTRAIN (NTRAINxNDIMS) - training data, one vector per row, PCA-transformed
    TDEV (NDEVxNDIMS)  - devtest data, one vector per row, PCA-transformed
    YTRAIN (NTRAIN) - true labels of each training vector
    YDEV (NDEV) - true labels of each devtest token
    VARIANCES - nonzero eigenvectors of the covariance matrix = eigenvectors of the gram matrix

    BESTSIZE (scalar) - the best size to use for the nearest-neighbor classifier
    ACCURACIES (NTRAIN) - accuracy of dev classification, as function of the size of the NN classifier

    The only sizes you need to test (the only nonzero entries in the ACCURACIES
    vector) are the ones where the PCA features explain between 92.5% and
    97.5% of the variance of the training set, as specified by the provided
    per-feature variances.  All others should be zero.
    '''

    pv = np.zeros_like(variances)

    for k in range(len(variances)):
        p_v = 100 * np.sum(variances[:k])/np.sum(variances)
        if p_v > 92.5 and p_v < 97.5:
            pv[k] = p_v
    best_size = np.nonzero(pv)[0]
    best_size += 1

    accuracies = np.zeros(len(ttrain))
    for i in range(len(best_size)):
        D = todo_distances(ttrain, tdev, best_size[i])
        hyps = todo_nearest_neighbor(Ytrain, D)
        accuracy, _ = todo_compute_accuracy(Ydev, hyps)
        accuracies[best_size[i]] = accuracy


    return best_size[0], accuracies
