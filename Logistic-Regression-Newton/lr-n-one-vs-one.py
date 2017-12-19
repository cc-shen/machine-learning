"""
  Name: Charles Shen
  Date: Tuesday October 31, 2017
"""

import numpy as np
import gc

def sigmoid(x):
  return 1. / (1. + np.exp(-x))

# Modify this hyper-tuning parameter as needed
Lambda = 0.05

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

  weighted_X_transposed = np.multiply(X, class_weights[:, np.newaxis]).T

  while iterations < max_iterations:
    p = sigmoid(np.dot(X, w))
    gradient = np.dot(weighted_X_transposed, p - (y + 1.)/2.) + 2. * Lambda * w
    hessian = np.dot(weighted_X_transposed, (X.T * (p * (1. - p))).T) + 2. * Lambda * np.eye(len(w))
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

# Training data
X_train, y_train = unpickle('cifar-10-batches-py/data_batch_1')

# Test data
X_test, y_test = unpickle('cifar-10-batches-py/test_batch')

# We're doing one-vs-one
total_classes = 10

print("Lambda:", Lambda)
print("")

intermediate_predictions = np.zeros((len(y_train), total_classes))

for i in range(total_classes):
  print("Class:", i)
  for j in range(i+1, total_classes):
    print("Sub-class:", j)
    y_train_modified = np.zeros(len(y_train))

    same_class_count = 0

    data_to_delete = []
    # Modifying our y_train
    for k in range(len(y_train)):
      if y_train[k] == i:
        y_train_modified[k] = 1
        same_class_count += 1
      elif y_train[k] == j:
        y_train_modified[k] = -1
      else:
        data_to_delete.append(k)

    # Modifying our training data
    X_train_modified = np.copy(X_train)
    X_train_modified = np.delete(X_train_modified, data_to_delete, 0)

    y_train_modified = np.delete(y_train_modified, data_to_delete, 0)

    class_weights = np.zeros(len(y_train_modified))
    same_class_weight = (1. / float(same_class_count))
    diff_class_weight = (1. / float(len(y_train_modified - same_class_count)))
    for k in range(len(y_train_modified)):
      # 1 is the class we're looking at
      if y_train_modified[k] == 1:
        class_weights[k] = same_class_weight
      else:
        class_weights[k] = diff_class_weight

    w = binary_logistic_regression(X_train_modified, y_train_modified, class_weights)

    predictions = sigmoid(np.dot(X_test, w))
    # Determine the class given the best predictions
    for k in range(len(predictions)):
      if predictions[k] >= 0.5:
        intermediate_predictions[k][i] += 1
      else:
        intermediate_predictions[k][j] += 1

  print("Done class:", i)
  print("")
  gc.collect()

real_predictions = np.zeros(len(y_test))
for i in range(len(intermediate_predictions)):
  # argmax to get the class
  real_predictions[i] = np.argmax(intermediate_predictions[i])

print("Test Error:", mean_square_error(real_predictions, y_test) * 100., "%")
