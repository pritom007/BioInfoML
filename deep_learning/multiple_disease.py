from __future__ import absolute_import, division, print_function
import re

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from data_processing import process_y
from sklearn.model_selection import train_test_split
from deep_learning import data_batch_handling as dbh

file_dir = r'C:\Users\Petros Debesay\PycharmProjects\BioInfoML\PCA'
import numpy as np


def return_x():
    my_data = np.genfromtxt(file_dir + r'\pca.csv', delimiter=',')
    return my_data


def return_y(key):
    my_data = np.array(process_y.return_y(key))
    return my_data


print(tf.__version__)

keys = process_y.return_dict().keys()

new_keys = set()


for key in keys:
    for word in re.compile('\w+').findall(key):
        new_keys.add(word)

features = len(return_x()[0])

# test_images = train_images[:1110]
# test_labels = train_labels[:1110]
#train_images = train_images  # [1110:]
#train_labels = train_labels  # [1110:]

# x = len(test_images) + len(train_images)
# print(x)

model = keras.Sequential([
    keras.layers.Dense(features, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    keras.layers.Dense(512, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    keras.layers.Dense(256, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    keras.layers.Dense(128, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    keras.layers.Dense(2, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l1(0.01))
])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])


def multiple_train(term):
    print('Term being searched is: ', term)

    x_train, x_test, y_train, y_test = train_test_split(return_x(), return_y(term), test_size=0.3)

    model.fit(x_train, y_train, epochs=300)
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print(len(x_test))
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)


for run in new_keys:
    multiple_train(run)
