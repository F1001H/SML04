import numpy as np
import matplotlib.pyplot as plt


def kernel(x_1, x_2, sigma=1):
    return float(np.exp((-1 / (2 * sigma ** 2)) * np.linalg.norm(x_1 - x_2) ** 2))


def compute_covariance_matrix(x_1, x_2):
    return np.array([[kernel(a, b)for a in x_1] for b in x_2])

def predict(x, y, values, noise=0.005,):
    inv_cov_matrix = np.linalg.inv(compute_covariance_matrix(x, x) + noise*np.identity(len(x)))
    k_1 = compute_covariance_matrix(x, values)
    k_2 = compute_covariance_matrix(values, values)
    mean = np.dot(k_1, np.dot(y, inv_cov_matrix.T).T).flatten()
    cov_matrix = k_2 - np.dot(k_1, np.dot(inv_cov_matrix, k_1.T))
    variance = np.diag(cov_matrix)
    return mean, variance

def plot_GP():
    x = np.arange(start=0, stop=2 * np.pi, step=0.005)
    y = [2 * np.cos(i) + np.sin(i) ** 2 for i in x]
    x_data = [x[0]]
    y_data = [y[0]]
    for i in range(2):
        x_new = np.array(x_data)
        y_new = np.array(y_data)
        mean, variance = predict(x_new, y_new, x)
        plt.plot(x, y, color="red")
        plt.plot(x, mean, color="blue")
        plt.fill_between(x, mean + 2 * np.sqrt(variance), mean - 2 * np.sqrt(variance), color="lightsteelblue")
        plt.scatter(x_new, y_new, color="black")
        plt.show()
        index = np.argmax(variance)
        x_data.append(x[index])
        y_data.append(y[index])
plot_GP()
