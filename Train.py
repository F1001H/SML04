from NeuralNetwork import NeuralNetwork
import numpy as np

test_data = np.loadtxt("data/mnist_small_test_in.txt")
test_labels = np.loadtxt("data/mnist_small_test_out.txt")
train_data = np.loadtxt("data/mnist_small_train_in.txt")
train_labels = np.loadtxt("data/mnist_small_train_out.txt")
train_labels = np.array(train_labels.astype('int32')[:, None] == np.arange(10), np.float64)
test_labels = np.array(test_labels.astype('int32')[:, None] == np.arange(10), np.float64)
neural_net = NeuralNetwork([784, 70, 10])
neural_net.train(train_data , train_labels, test_data, test_labels, 10, 0.05, 40)

