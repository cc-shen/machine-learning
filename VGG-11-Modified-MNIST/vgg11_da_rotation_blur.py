"""
Name: Charles Shen
Date: 2017-11-21
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt


# MNIST settings
num_classes = 10
img_rows = 28
img_cols = 28

# Loading MNIST data
(X_train, y_train), (X_test_org, y_test) = mnist.load_data()

# Training data augmentation
n = len(X_train)
aug_X_train = []
aug_y_train = []
for i in range(n):
    aug_X_train.append(np.asarray(Image.fromarray(X_train[i].reshape((img_rows, img_cols)) * 255.).convert('L').filter(ImageFilter.GaussianBlur(1))))
    aug_y_train.append(y_train[i])

X_train = np.append(X_train, aug_X_train, axis=0)
y_train = np.append(y_train, aug_y_train)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1).astype('float32') / 255.  # Normalize (max value is 255)

y_train = keras.utils.to_categorical(y_train, num_classes)

X_test = X_test_org.reshape(X_test_org.shape[0], img_rows, img_cols, 1).astype('float32') / 255.  # Normalize (max value is 255)

y_test = keras.utils.to_categorical(y_test, num_classes)


def X_test_data_augment(blur=0, rotation_degree=0):
    augmented = []
    for x_test in X_test_org:
        augmented.append(Image.fromarray(x_test.reshape((img_rows, img_cols)) * 255.).convert('L').filter(ImageFilter.GaussianBlur(blur)).rotate(rotation_degree))

    augmented_X_test = []
    for aug in augmented:
        augmented_X_test.append(np.asarray(aug))

    augmented_X_test = np.asarray(augmented_X_test)
    augmented_X_test = augmented_X_test.reshape(augmented_X_test.shape[0], img_rows, img_cols, 1).astype('float32') / 225.

    return augmented_X_test


# Our VGG11 model, model constructed as suggested in keras tutorial
#   (https://keras.io/getting-started/sequential-model-guide/)
# Along with some modification as explained in the pdf response
def VGG11_Model(epo=5, batch_size=32):
    print("epochs:", epo)
    print("batch size:", batch_size)
    print("Training start...")
    print("")

    model = Sequential()
    # Reshaping from 28x28 to 32x32 for VGG
    model.add(ZeroPadding2D((2, 2), input_shape=(img_rows, img_cols, 1)))
    # Since we have added padding of 2 on each side
    model.add(Conv2D(64, (3, 3), activation='relu',
                     input_shape=(img_rows + 4, img_cols + 4, 1),
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    # This causes the model to not converge... Most likely due to the
    # small size of MNIST dataset which pools into 2 by 2 and then pad
    # it again (i.e. losing data)
    # model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    # This causes the model to not converge as well
    #   Reason is the same as above (loss of data if max pooling)
    # model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Flatten())
    # We have smaller images so the 4096 channels mentioned in the paper is
    # too much for our purposes here
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    # This layer has been removed as I found it did not affect accuracy
    # model.add(Dense(1024, activation='relu'))
    # Instead of 1000, I use 10 because MNIST only have 10 classes
    # However, including this layer causes the model to over-fit after 3 epochs
    # model.add(Dense(num_classes, activation='relu'))  # we have 10 classes
    # model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # we have 10 classes

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    print(model.summary())
    print("")

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epo,
              verbose=1,
              validation_data=(X_test, y_test))

    print("Training done!")
    return model


# For rotating
model = VGG11_Model(epo=5, batch_size=32)

rotated_test_accuracy = []
rotations = [-45., -40., -35., -30., -25., -20., -15., -10., -5., 0., 5., 10., 15., 20., 25., 30., 35., 40., 45.]
for rotation in rotations:
    print("degree of rotation:", rotation)
    score = model.evaluate(X_test_data_augment(rotation_degree=rotation),
                           y_test, batch_size=32, verbose=1)
    rotated_test_accuracy.append(score[1])
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print("")

plt.scatter(rotations, rotated_test_accuracy)
plt.title("test accuracy vs degree of rotation")
plt.ylabel("test accuracy")
plt.xlabel("degree of rotation")
plt.show()

del model

# For blurring
model = VGG11_Model(epo=5, batch_size=32)

blurred_test_accuracy = []
blurs = [0., 1., 2., 3., 4., 5., 6.]
for blur in blurs:
    print("blur:", blur)
    score = model.evaluate(X_test_data_augment(blur=blur),
                           y_test, batch_size=32, verbose=1)
    blurred_test_accuracy.append(score[1])
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print("")

plt.scatter(blurs, blurred_test_accuracy)
plt.title("test accuracy vs blur")
plt.ylabel("test accuracy")
plt.xlabel("blur")
plt.show()
