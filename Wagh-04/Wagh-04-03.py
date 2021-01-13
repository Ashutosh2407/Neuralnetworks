# Wagh, Ashutosh
# 1001-522-863
# 2020_04_19
# Assignment-04-03

#import pytest
import numpy as np
from cnn import CNN
import os
import imp
from tensorflow import keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import pytest

def test_train():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
    y_train = y_train.flatten().astype(np.int32)
    X_test = X_test.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
    y_test = y_test.flatten().astype(np.int32)
    input_dimension = X_train.shape[1]

    model = CNN()
    model.add_input_layer(shape=input_dimension,name="input0")

    model.append_dense_layer(num_nodes=32,activation='relu',name="dense1")
    weights = model.get_weights_without_biases(layer_name="dense1")
    np.random.seed(seed=2)
    weights1= np.random.randn(*weights.shape)
    weights1 = weights1.reshape(weights.shape)
    model.set_weights_without_biases(weights = weights1,layer_name="dense1")
    model.append_dense_layer(num_nodes=16,activation='relu',name="dense2")
    weights = model.get_weights_without_biases(layer_name="dense2")
    weights1= np.random.randn(*weights.shape)
    weights1 = weights1.reshape(weights.shape)
    model.set_weights_without_biases(weights = weights1,layer_name="dense2")
    model.append_dense_layer(num_nodes=10,activation='softmax',name="dense3")
    weights = model.get_weights_without_biases(layer_name="dense3")
    weights1= np.random.randn(*weights.shape)
    weights1 = weights1.reshape(weights.shape)
    model.set_weights_without_biases(weights = weights1,layer_name="dense3")
    model.set_loss_function(loss="SparseCategoricalCrossentropy")
    model.set_optimizer(optimizer="SGD")
    model.set_metric(['accuracy'])
    accuracy = model.train(X_train,y_train,128,5)
    print(accuracy)
    np.testing.assert_almost_equal(np.array(accuracy),np.array([4.8,2.3,2.3,2.3,2.3]),decimal=1)

def test_evaluate():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
    y_train = y_train.flatten().astype(np.int32)
    X_test = X_test.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
    y_test = y_test.flatten().astype(np.int32)
    input_dimension = X_train.shape[1]

    model = CNN()
    model.add_input_layer(shape=input_dimension,name="input0")

    model.append_dense_layer(num_nodes=32,activation='relu',name="dense1")
    weights = model.get_weights_without_biases(layer_name="dense1")
    np.random.seed(seed=2)
    weights1= np.random.randn(*weights.shape)
    weights1 = weights1.reshape(weights.shape)
    model.set_weights_without_biases(weights = weights1,layer_name="dense1")
    model.append_dense_layer(num_nodes=16,activation='relu',name="dense2")
    weights = model.get_weights_without_biases(layer_name="dense2")
    weights1= np.random.randn(*weights.shape)
    weights1 = weights1.reshape(weights.shape)
    model.set_weights_without_biases(weights = weights1,layer_name="dense2")
    model.append_dense_layer(num_nodes=10,activation='softmax',name="dense3")
    weights = model.get_weights_without_biases(layer_name="dense3")
    weights1= np.random.randn(*weights.shape)
    weights1 = weights1.reshape(weights.shape)
    model.set_weights_without_biases(weights = weights1,layer_name="dense3")
    model.set_loss_function(loss="SparseCategoricalCrossentropy")
    model.set_optimizer(optimizer="SGD")
    model.set_metric(['accuracy'])
    loss = model.train(X_train,y_train,128,5)
    evaluate = model.evaluate(X_test,y_test)
    print(evaluate)
    np.testing.assert_almost_equal(np.array(evaluate),np.array([2.29,0.128]),decimal=1)
