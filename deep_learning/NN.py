from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from data_processing import process_y

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


print(len(return_x()))
print(return_y('acute lymphoblastic leukemia'))


print(tf.__version__)

features = 100  # as defined by pca
data_batch_handling.remove()
(train_images, train_labels), (test_images, test_labels) = (tst.get_train_data()), (tst.get_test_data())

print(train_images.shape)
#print(train_labels)

model = keras.Sequential([
    keras.layers.Dense(features),
    keras.layers.Dense(128, activation=tf.nn.sigmoid),
    keras.layers.Dense(2, activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=150)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)