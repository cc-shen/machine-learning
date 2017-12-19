"""
    Charles Shen
    October 11, 2017
"""

# Lin. alg. calculations
import numpy
# Distance calcuations
import scipy.spatial
# MNIST data set
import keras
from keras.datasets import mnist


# || x - y ||_{2}
def euclidean_distance(x, y):
    return scipy.spatial.distance.euclidean(x, y)


# || x - y ||_{1}
def manhattan_distance(x, y):
    return scipy.spatial.distance.cityblock(x, y)


# || x - y ||_{inf}
def chebyshev_distance(x, y):
    return scipy.spatial.distance.chebyshev(x, y)


def k_nn(distances, y, k):
    """
      argpartition is a O(n) time complexity as it uses 'introselect'
          which is a O(n) worst case selection algorithm, this returns the indices
      See https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.argpartition.html
          and https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.partition.html#numpy.partition
    """
    k_smallest = numpy.argpartition(distances, range(k))[:k]
    return numpy.bincount(y[k_smallest]).argmax()


def find_errors(predictions, results):
    errors = 0
    for i in range(len(predictions)):
        if predictions[i] != results[i]:
            errors += 1
    return float(errors) / len(results)


cv_fold = 10
max_k = 15


def find_best_k(X_train, y_train):
    print("Finding the best k...")
    # Cross-Validation, cv_fold-fold (10 here)
    set_size = int(len(X_train) / cv_fold)
    errors_rate = numpy.zeros((max_k, cv_fold))
    for fold in range(cv_fold):
        # Take the fold-th training set
        X_train_i = numpy.delete(X_train, range(fold * set_size, (fold + 1) * set_size), axis=0)
        y_train_i = numpy.delete(y_train, range(fold * set_size, (fold + 1) * set_size), axis=0)

        # Take the set to validate against
        X_test_i = X_train[(fold) * set_size:(fold + 1) * set_size, :]
        y_test_i = y_train[(fold) * set_size:(fold + 1) * set_size]

        # Find the distance between the training X and training y
        # Cannot do the entire training set due to memory constraints
        relative_dist = scipy.spatial.distance.cdist(X_train_i, X_test_i, 'euclidean')

        m = len(X_test_i)
        predictions = numpy.zeros((max_k, m))
        for t in range(m):
            for k in range(max_k):
                predictions[k][t] = k_nn(relative_dist[:,t], y_train_i, k + 1)
        for k in range(max_k):
            errors_rate[k][fold] = find_errors(predictions[k], y_test_i)
    mean_errors_rate = list(range(max_k))
    for k in range(max_k):
        mean_errors_rate[k] = numpy.mean(errors_rate[k])
        print("k:", k + 1)
        print("Error:", mean_errors_rate[k])
        print("")
    smallest_k = numpy.argpartition(mean_errors_rate, range(1))[:1]
    # Need to +1 because we get an index that's 0-based
    print("Smallest k:", smallest_k[0] + 1)
    return (smallest_k[0] + 1)


def run_test(X_train, y_train, X_test, y_test, k):
    print("Running test data... Using k =", k)
    print("Calculating relative distances...")
    relative_dist = scipy.spatial.distance.cdist(X_train, X_test, 'euclidean')
    print("Calculations completed")
    m = len(X_test)
    predictions = numpy.zeros(m)
    for t in range(m):
        predictions[t] = k_nn(relative_dist[:, t], y_train, k)
    print("Error rate on test data:", find_errors(predictions, y_test))
    print("Done!")


(X_train, y_train), (X_test, y_test) = mnist.load_data()

k = find_best_k(X_train, y_train)
run_test(X_train, y_train, X_test, y_test, k)
