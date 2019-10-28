from __future__ import absolute_import, division, print_function

# Helper libraries
import matplotlib.pyplot as plt
# TensorFlow and tf.keras
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras

from data_processing import process_y

file_dir = r'D:\PycharmProjects\BioInfoML\PCA'

# Helper libraries
import numpy as np


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def return_x():
    my_data = np.genfromtxt(file_dir + r'\pca_dl.csv', delimiter=',')

    #my_data = np.genfromtxt(file_dir + r'\processed.csv', delimiter=',')

    return my_data


def return_y(key):
    my_data = np.array(process_y.return_y(key))
    return my_data


print(tf.__version__)
# bone marrow
name_disease = "breast"
# x_set = return_x()[1:]
x_train, x_test, y_train, y_test = train_test_split(return_x(), return_y(name_disease), test_size=0.3)
features = len(x_train[0])

model = keras.Sequential([
    keras.layers.Dense(features, kernel_regularizer=tf.keras.regularizers.l1(0.1)),
    keras.layers.Dense(256, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1(0.1)),
    keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l1(0.1)),
    keras.layers.Dense(2, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l1(0.1))
])
# metrics=[keras_metrics.precision(), keras_metrics.recall()])


model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['acc', f1_m, precision_m, recall_m])

history = model.fit(x_train, y_train, batch_size=88, epochs=15)
loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
y_predict = model.predict(x_test)
print(f'loss: {loss}, acc: {accuracy}, f1_score: {f1_score}, precision: {precision}, recall: {recall}')
# print(classification_report(y_test, y_res))
# print('Test accuracy:', test_acc)
print('Diesase:' + name_disease + " tumor")
plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('model accuracy for ' + name_disease + " tumor")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig(name_disease + "_acc.png")
# summarize history for loss
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss ' + name_disease + " tumor")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig(name_disease + "_loss.png")

file = open("reg_disease.txt", 'a')

file.write(
    name_disease + f' loss: {loss}, acc: {accuracy}, f1_score: {f1_score}, precision: {precision}, recall: {recall}\n')
file.write("regularization: 0.1\n")
file.write(classification_report(y_test, y_predict) + '\n')
file.write("=======================" + '\n')
file.close()
