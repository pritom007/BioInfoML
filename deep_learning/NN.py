from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from data_processing import process_y
from sklearn.model_selection import train_test_split
from deep_learning import data_batch_handling as dbh

file_dir = r'C:\Users\Petros Debesay\PycharmProjects\BioInfoML\PCA'

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


def return_x():
    my_data = np.genfromtxt(file_dir + r'\pca.csv', delimiter=',')
    return my_data


def return_y(key):
    my_data = np.array(process_y.return_y(key))
    return my_data


print(tf.__version__)

x_train, x_test, y_train, y_test = train_test_split(return_x(), return_y('leukemia'), test_size=0.3)
features = len(x_train[0])

model = keras.Sequential([
    keras.layers.Dense(features, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    keras.layers.Dense(512, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    keras.layers.Dense(256, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    keras.layers.Dense(128, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    keras.layers.Dense(2, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l1(0.01))
])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

model.fit(x_train, y_train, epochs=100)
test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)
