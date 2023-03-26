import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
import os
from keras.utils import np_utils
from keras.layers import Conv2D, Flatten

(train_x, train_y), (test_x, test_y) = cifar10.load_data()

num_classes = 10
train_y = np_utils.to_categorical(train_y)
test_y = np_utils.to_categorical(test_y, num_classes)

train_x = train_x.astype('float32') # this is necessary for the division below
train_x /= 255
test_x = test_x.astype('float32') / 255

img_rows = img_cols = 32
channels = 3

train_x_reshaped = train_x.reshape(len(train_x), img_rows, img_cols, channels)
test_x_reshaped = test_x.reshape(len(test_x), img_rows, img_cols, channels)


simple_cnn_model = Sequential()
simple_cnn_model.add(Conv2D(32, (3,3), input_shape=(img_rows,img_cols,channels), activation='relu'))
simple_cnn_model.add(Conv2D(32, (3,3), activation='relu'))
simple_cnn_model.add(Conv2D(32, (3,3), activation='relu'))
simple_cnn_model.add(Flatten())
simple_cnn_model.add(Dense(10, activation='softmax'))

simple_cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
simple_cnn_model_history = simple_cnn_model.fit(train_x_reshaped, train_y, batch_size=100, epochs=10, validation_data=(test_x_reshaped, test_y))



# Saving the model for Future Inferences

model_json = simple_cnn_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
simple_cnn_model.save_weights("model.h5")

