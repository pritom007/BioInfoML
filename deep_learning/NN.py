from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from data_processing import process_y
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

(train_images, train_labels) = (return_x(), return_y('leukemia'))
features = len(train_images[0])


test_images = train_images[:1110]
test_labels = train_labels[:1110]
train_images = train_images[1110:]
train_labels = train_labels[1110:]


x = len(test_images) + len(train_images)
print(x)


model = keras.Sequential([
    keras.layers.Dense(features, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    keras.layers.Dense(512, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    keras.layers.Dense(256, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    keras.layers.Dense(128, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l1(0.01)),
    keras.layers.Dense(2, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l1(0.01))
])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=100)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
