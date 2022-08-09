import numpy as np

def accuracy(y, out):
    return np.mean(np.argmax(y, axis=-1) == np.argmax(out, axis=-1))

def softmax(x: np.ndarray) -> float:
    sm = np.exp(x - np.max(x))
    return sm / np.sum(sm, axis=0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self, nodes: list):
        self.cache = {}
        self.parameters = {
            'w1': np.random.rand(nodes[1], nodes[0]) * np.sqrt(1.0 / nodes[0]),
            'w2': np.random.rand(nodes[2], nodes[1]) * np.sqrt(1.0 / nodes[0]),
            'b1': np.zeros((nodes[1], 1)) * np.sqrt(1.0 / nodes[0]),
            'b2': np.zeros((nodes[2], 1)) * np.sqrt(1.0 / nodes[0])
        }

    def feed_forward(self, x):
        self.cache['x'] = x
        self.cache['o1'] = np.matmul(self.parameters['w1'], self.cache['x'].T) + self.parameters['b1']
        self.cache['f1'] = sigmoid(self.cache['o1'])
        self.cache['o2'] = np.matmul(self.parameters['w2'], self.cache['f1']) + self.parameters['b2']
        self.cache['f2'] = softmax(self.cache['o2'])
        return self.cache['f2']

    def back_propagate(self, y, out):
        delta_o2 = out - y.T
        batch_size = y.shape[0]
        delta_w2 = (1.0/batch_size) * np.matmul(delta_o2, self.cache['o1'].T)
        delta_b2 = (1.0/batch_size) * np.sum(delta_o2, axis=1, keepdims=True)
        delta_f1 = np.matmul(self.parameters['w2'].T, delta_o2)
        delta_o1 = delta_f1*sigmoid_derivative(self.cache['o1'])
        delta_w1 = (1.0/batch_size) * np.matmul(delta_o1, self.cache['x'])
        delta_b1 = (1.0/batch_size) * np.sum(delta_o1, axis=1, keepdims=True)
        return {'w1': delta_w1, 'w2': delta_w2, 'b1': delta_b1, 'b2': delta_b2}

    def train(self, train_x, train_y, test_x, test_y, epochs, learning_rate, batch_size):
        num_batches = -(-train_x.shape[0] // batch_size)
        for i in range(epochs):
            random_permutation = np.random.permutation(train_x.shape[0])
            train_x_permutation = train_x[random_permutation]
            train_y_permutation = train_y[random_permutation]
            for j in range(num_batches):
                start = j * batch_size
                end = min(start + batch_size, train_x.shape[0])
                x = train_x_permutation[start:end]
                y = train_y_permutation[start:end]
                out = self.feed_forward(x)
                gradients = self.back_propagate(y, out)
                for k in self.parameters:
                    self.parameters[k] = self.parameters[k] - learning_rate * gradients[k]
        out = self.feed_forward(train_x)
        print(out[: , 6000])