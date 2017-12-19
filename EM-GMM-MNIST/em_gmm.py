"""
  Name: Charles Shen
  Date: Sunday November 12, 2017
"""

# Lin algebra operations
import numpy as np
# Forcing garbage collections at times
import gc
# Reduce dimension via PCA (as suggested in the problem)
from sklearn.decomposition import PCA
# MNIST data
import keras
from keras.dataset import mnist

# For timing how long the operations take
import time

# Exercise 1 Problem 1 Section
# Calculate the Gaussian
# X is O(nd)
# mu is O(d)
# Sigma is O(d)
# Overall run time is $ O(nd)
def gaussian(X, mu, Sigma):
  # In case we get divide by 0 issues
  Sigma = Sigma + 0.00000001
  # Sigma is a diagonal matrix (but stored as a vector)
  det_S = np.prod(Sigma) # O(d) time, O(1) space
  inv_S = 1 / Sigma # O(d) time and space
  variance = (X - mu)**2 # O(nd)
  return (det_S**(-0.5)) * np.exp(-0.5 * np.dot(inv_S, variance.T)) # O(nd) time, O(n) space

# K is the number of Gaussian models
# Runs in O(dnK) time and O(max(nK, Kd)) space
def GMM(X, max_iterations=500, K=5, tolerance=10**-5):
  n, d = X.shape

  # random initialization
  mean = X[np.random.choice(n, K, False), :] # O(Kd) time and space
  covariances = np.ones((K, d)) # O(Kd) time and space

  weight = np.ones(K) / float(K) # O(K) time and space

  R = np.zeros((n, K)) # O(nK) time and space
  all_log_likelihoods = []

  while len(all_log_likelihoods) < max_iterations:
    # O(Knd) time
    for k in range(K):
      R[:,k] = weight[k] * gaussian(X, mean[k], covariances[k])

    log_likelihood = -np.sum(np.log(np.sum(R, axis=1))) # O(nK) time and O(1) space
    # print(log_likelihood)
    all_log_likelihoods.append(log_likelihood)

    # Check for log likelihood convergence
    if len(all_log_likelihoods) > 1 and np.abs(log_likelihood - all_log_likelihoods[-2]) < (tolerance * log_likelihood):
      break

    R = (R.T / np.sum(R, axis=1)).T # O(nK) time and space
    r_k = np.sum(R, axis=0) # O(nK) time and O(K) space
    weight = r_k / float(n) # O(K) time and space
    mean = (np.dot(X.T, R) / r_k).T # O(dnK) time and O(Kd) space
    covariances = (np.dot((X**2).T, R) / r_k).T - (mean**2) # O(dnK) time and O(Kd) space

  return mean, covariances, weight

# Exercise 1 Problem 2 Section
def make_prediction(X, mean, Sigma, weight):
  probability = np.array([weight_i * gaussian(X, mu_i, sigma_i) for mu_i, sigma_i, weight_i in zip(mean, Sigma, weight)])
  return np.sum(probability)

def mean_square_error(predictions, answers):
  return np.count_nonzero(predictions - answers) / float(len(answers))

time_start = time.time()
# This data set has 10 classes
num_classes = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Data normalization
X_train = (1. / float(np.amax(np.absolute(X_train)))) * X_train
X_test = (1. / float(np.amax(np.absolute(X_test)))) * X_test

load_time = time.time() - time_start
print("Data took", time.time() - time_start, "seconds to load.")
print("")

# Reduce dimensions via PCA
num_gaussians = 5
print("Number of Gaussian:", num_gaussians)
print("")
pca = PCA(n_components=10).fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

print("PCA took", time.time() - time_start - load_time, "seconds to reduce dimensions.")
print("")

# Categorized the points corresponding to its class, with 10 classes possible
categorized_X_train = []
for c in range(num_classes):
  categorized_X_train.append([])
for i in range(len(y_train)):
  categorized_X_train[y_train[i]].append(X_train[i])

int_probabilties = np.zeros((num_classes, len(y_test)))
for i in range(num_classes):
  time_start_i = time.time()
  print("Class:", i)
  # Fitting our model
  mean, covariances, weight = GMM(np.array(categorized_X_train[i]), K=num_gaussians)

  weighted_class_i = len(categorized_X_train[i]) / float(len(X_train))
  # Make predictions given our model for class i
  for j in range(len(X_test)):
    int_probabilties[i][j] = weighted_class_i * make_prediction(X_test[j], mean, covariances, weight)

  print("Done class", i, "after", time.time() - time_start_i, "seconds.")
  print("")
  gc.collect()

# For each point, get the class that had the highest probability
predictions = np.zeros(len(y_test))
for i in range(len(predictions)):
  predictions[i] = int_probabilties[:,i].argmax()

# Compare predictions
print("Error rate:", mean_square_error(predictions, y_test) * 100., "%")
print("Finished after", time.time() - time_start, "seconds.")
