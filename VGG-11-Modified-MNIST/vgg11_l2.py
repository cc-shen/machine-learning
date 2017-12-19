"""
Name: Charles Shen
Date: 2017-11-21
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.regularizers import l2
import matplotlib.pyplot as plt


# Recording accuracy and losses
class AccuracyAndLossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.validation_losses = []
        self.validation_accuracy = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))
        self.validation_losses.append(logs.get('val_loss'))
        self.validation_accuracy.append(logs.get('val_acc'))


# MNIST settings
num_classes = 10
img_rows = 28
img_cols = 28

# Loading MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1).astype('float32') / 255.  # Normalize (max value is 255)

y_train = keras.utils.to_categorical(y_train, num_classes)

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1).astype('float32') / 255.  # Normalize (max value is 255)

y_test = keras.utils.to_categorical(y_test, num_classes)


# Our VGG11 model, model constructed as suggested in keras tutorial
#   (https://keras.io/getting-started/sequential-model-guide/)
# Along with some modification as explained in the pdf response
def VGG11_L2(epo=7, batch_size=32):
    print("epochs:", epo)
    print("batch size:", batch_size)
    print("Training start...")
    print("")

    # With l2 regularization
    l2_regularizer = l2(5 * (10 ** -4))

    model = Sequential()
    # Reshaping from 28x28 to 32x32 for VGG
    model.add(ZeroPadding2D((2, 2), input_shape=(img_rows, img_cols, 1)))
    # Since we have added padding of 2 on each side
    model.add(Conv2D(64, (3, 3), activation='relu',
                     input_shape=(img_rows + 4, img_cols + 4, 1),
                     padding='same',
                     kernel_regularizer=l2_regularizer))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2_regularizer))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2_regularizer))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2_regularizer))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2_regularizer))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2_regularizer))
    # This causes the model to not converge... Most likely due to the
    # small size of MNIST dataset which pools into 2 by 2 and then pad
    # it again (i.e. losing data)
    # model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2_regularizer))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2_regularizer))
    # This causes the model to not converge as well
    #   Reason is the same as above (loss of data if max pooling)
    # model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Flatten())
    # We have smaller images so the 4096 channels mentioned in the paper is
    # too much for our purposes here
    model.add(Dense(1024, activation='relu',
                    kernel_regularizer=l2_regularizer))
    model.add(Dropout(0.5))
    # This layer has been removed as I found it did not affect accuracy
    # model.add(Dense(1024, activation='relu'))
    # Instead of 1000, I use 10 because MNIST only have 10 classes
    # However, including this layer causes the model to over-fit after 3 epochs
    # model.add(Dense(num_classes, activation='relu'))  # we have 10 classes
    # model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax',
                    kernel_regularizer=l2_regularizer))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    print(model.summary())
    print("")

    history = AccuracyAndLossHistory()
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epo,
              verbose=1,
              validation_data=(X_test, y_test),
              callbacks=[history])

    # Saving the trained model helped me do the other parts of this exercise faster
    model.save('vgg11_l2.h5')

    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    del model
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print("")
    return history


history = VGG11_L2(epo=7, batch_size=32)

# Plotting test accuracy vs number of iterations
plt.scatter(list(range(len(history.validation_accuracy))),
            history.validation_accuracy)
plt.title("test accuracy vs number of iterations")
plt.ylabel("test accuracy")
plt.xlabel("number of iterations")
plt.show()

# Plotting training accuracy vs number of iterations
plt.scatter(list(range(len(history.accuracy))),
            history.accuracy)
plt.title("training accuracy vs number of iterations")
plt.ylabel("training accuracy")
plt.xlabel("number of iterations")
plt.show()

# Plotting test loss vs number of iterations
plt.scatter(list(range(len(history.validation_losses))),
            history.validation_losses)
plt.title("test loss vs number of iterations")
plt.ylabel("test loss")
plt.xlabel("number of iterations")
plt.show()

# Plotting training loss vs number of iterations
plt.scatter(list(range(len(history.losses))),
            history.losses)
plt.title("training loss vs number of iterations")
plt.ylabel("training loss")
plt.xlabel("number of iterations")
plt.show()
