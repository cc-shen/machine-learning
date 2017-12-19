"""
  Name: Charles Shen
  Date: Saturday October 28, 2017
"""

import numpy as np
import gc

def sigmoid(x):
  return 1. / (1. + np.exp(-x))

# Modify this hyper-tuning parameter as needed
Lambda = 0.001

# Tolerance
tolerance = 0.0001
# Max iterations
max_iterations = 50

# We need to normalize the two binary classes which is in class_weights
# It contains the value of (1 / n_{+}) or (1 / n_{-}) depending on the what class the
#   data point is in
def binary_logistic_regression(X, y, class_weights):
  w = np.zeros(len(X[0]))
  new_w = np.zeros(len(X[0]))
  iterations = 0

  while iterations < max_iterations:
    p = sigmoid(np.dot(X, w))
    gradient = np.dot(X.T, class_weights * (p - (y + 1.)/2.)) + 2. * Lambda * w
    hessian = np.dot(X.T, (X.T * (class_weights * (p * (1. - p)))).T) + 2. * Lambda * np.eye(len(w))
    new_w = w - np.linalg.solve(hessian, gradient)
    iterations += 1

    if (np.linalg.norm(w - new_w, ord=2)) <= tolerance:
      break

    w = np.copy(new_w)

  return new_w

def mean_square_error(predictions, answers):
  return np.count_nonzero(predictions - answers) / float(len(answers))

# Loading CIFAR-10 data as suggested on the website
def unpickle(filename):
  import pickle
  with open(filename, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  X = np.asarray(dict[b'data'])
  # Adding bias
  X = np.c_[X, np.ones(len(X))]
  # Normalizing X_train
  X = (1. / float(np.amax(np.absolute(X)))) * X

  y = np.asarray(dict[b'labels'])
  return X, y

X_train, y_train = unpickle('cifar-10-batches-py/data_batch_1')

# We're doing one-vs-all
total_classes = 10

print("Lambda:", Lambda)
print("")

weights = []
for i in range(total_classes):
  print("Class:", i)

  y_train_modified = np.zeros(len(y_train))

  same_class_count = 0

  for j in range(len(y_train)):
    if y_train[j] == i:
      y_train_modified[j] = 1
      same_class_count += 1
    else:
      y_train_modified[j] = -1

  class_weights = np.zeros(len(y_train))
  same_class_weight = (1. / float(same_class_count))
  diff_class_weight = (1. / float(len(y_train - same_class_count)))
  for j in range(len(y_train)):
    if y_train[j] == i:
      class_weights[j] = same_class_weight
    else:
      class_weights[j] = diff_class_weight

  weights.append(binary_logistic_regression(X_train, y_train_modified, class_weights))
  print("Done class", i)
  print("")

  gc.collect()

X_test, y_test = unpickle('cifar-10-batches-py/test_batch')

real_predictions = np.zeros(len(y_test))
confidence = np.zeros(len(y_test))
for i in range(len(weights)):
  predictions = sigmoid(np.dot(X_test, weights[i]))
  # Determine the class given the best predictions
  for j in range(len(predictions)):
    if confidence[j] < predictions[j]:
      confidence[j] = predictions[j]
      real_predictions[j] = i

  gc.collect()

print("Test Error:", mean_square_error(real_predictions, y_test) * 100., "%")
