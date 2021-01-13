# Wagh, Ashutosh
# 1001-522-863
# 2020-03-22
# Assignment-03-01

# %tensorflow_version 2.x
# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimension = input_dimension
        self.weights = []
        self.bias = []
        self.transfer_function = []

    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        self.transfer_function.append(transfer_function.lower())
        if len(self.weights) == 0:
            self.weights.append(np.random.randn(self.input_dimension,num_nodes))
        else:
            self.weights.append(np.random.randn(len(self.weights[-1][0]), num_nodes))
        self.bias.append(np.random.randn(1,num_nodes))



    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        return self.weights[layer_number]

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        return self.bias[layer_number]

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.weights[layer_number] = weights
        

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.bias[layer_number]=biases

    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_hat))

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        for layer in range(len(self.weights)):
            X = tf.matmul(X, self.weights[layer]) + self.bias[layer]
            if self.transfer_function[layer] == 'relu':
                X = tf.nn.relu(X)
            elif self.transfer_function[layer] == 'sigmoid':
                X = tf.math.sigmoid(X)
        return X

        
    

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        data = dataset.batch(batch_size)
        for _ in range(num_epochs):
            for features, labels in data:
                with tf.GradientTape() as tape:
                    y_hat= self.predict(features)
                    loss = self.calculate_loss(labels,y_hat)
                dw, db = tape.gradient(loss, [self.weights, self.bias])
                for i in range(len(self.weights)):
                    self.weights[i].assign_sub(alpha * dw[i])
                    self.bias[i].assign_sub(alpha * db[i])

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        y_hat = tf.argmax(self.predict(X), 1)
        err = tf.math.not_equal(y_hat, y)
        percent_error = tf.math.count_nonzero(err)/X.shape[0]
        return percent_error

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        confusion_matrix = np.zeros((self.bias[-1].shape[1], self.bias[-1].shape[1]), dtype=int)
        Y = self.predict(X)
        Y = tf.argmax(Y, 1)
        for i in range(y.shape[0]):
            confusion_matrix[y[i], Y[i]] += 1
        return confusion_matrix
