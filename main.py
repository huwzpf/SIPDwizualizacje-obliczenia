import copy
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv


class NeuralNet:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return NeuralNet.sigmoid(x) * (1 - NeuralNet.sigmoid(x))

    @staticmethod
    def relu(x):
        return x * (x > 0)

    @staticmethod
    def relu_derivative(x):
        return x > 0

    @staticmethod
    def logistic_loss(h, y):
        return y * np.log(h) + (1 - y) * np.log(1 - h)

    class ActivationType(Enum):
        RELU = 0,
        SIGMOID = 1

    def __init__(self, input_size, layers_layout, output_size, activation_functions=None):
        layers_layout.append(output_size)
        self.layers_count = len(layers_layout)

        self.k = output_size

        if activation_functions is None:
            self.activation_functions = [NeuralNet.ActivationType.SIGMOID for _ in range(self.layers_count)]
        else:
            if len(activation_functions) != self.layers_count:
                raise IndexError("incorrect size")
            self.activation_functions = activation_functions

        self.z = [np.empty(1) for _ in range(self.layers_count)]
        self.activations = [np.empty(1) for _ in range(self.layers_count)]
        self.biases = [np.zeros((1, layer_size)) for layer_size in layers_layout]
        self.weights = []

        self.initialize_weights(input_size, layers_layout)

    def initialize_weights(self, input_size, layers_layout):
        print("WEIGHTS AFTER INIT")
        cnt = 0
        prev_layer_size = input_size
        for layer_size in layers_layout:
            self.weights.append(np.random.randn(prev_layer_size, layer_size) * np.sqrt(1/prev_layer_size))
            # self.weights.append(np.zeros((prev_layer_size, layer_size)))
            cnt += 1
            print(f"layer {cnt} :")
            print(self.weights[-1])
            prev_layer_size = layer_size



    def activation(self, x, layer):
        if self.activation_functions[layer] == NeuralNet.ActivationType.RELU:
            return self.relu(x)
        else:
            return self.sigmoid(x)

    def activation_derivative(self, x, layer):
        if self.activation_functions[layer] == NeuralNet.ActivationType.RELU:
            return self.relu_derivative(x)
        else:
            return self.sigmoid_derivative(x)

    def cost_function(self, x, y):
        self.forward_prop(x)
        # sum of logistic losses for all outputs
        cost = sum([NeuralNet.logistic_loss(a, b) for a, b in zip(np.nditer(self.activations[-1]), np.nditer(y))])
        return (-cost)/y.shape[0]

    def forward_prop(self, x):
        self.z[0] = x.dot(self.weights[0]) + self.biases[0]
        self.activations[0] = self.activation(self.z[0], 0)
        for i in range(1, self.layers_count):
            self.z[i] = self.activations[i-1].dot(self.weights[i]) + self.biases[i]
            self.activations[i] = self.activation(self.z[i], i)

    def train_batch(self, x, y, learning_rate):
        self.forward_prop(x)
        batch_size = y.shape[0]
        deltas = [0 for _ in range(self.layers_count)]
        deltas[-1] = self.activations[-1] - y
        for i in range(self.layers_count - 2, -1, -1):
            deltas[i] = np.transpose(self.weights[i + 1].dot(np.transpose(deltas[i + 1])))
            deltas[i] = np.multiply(deltas[i], self.activation_derivative(self.z[i], i))
        self.weights[0] -= np.transpose(x).dot(deltas[0]) * learning_rate/batch_size
        self.biases[0] -= np.sum(deltas[0], axis=0).reshape(self.biases[0].shape) * learning_rate/batch_size
        for i in range(1, self.layers_count):
            dw = np.transpose(self.activations[i-1]).dot(deltas[i])
            self.weights[i] -= dw * learning_rate/batch_size
            db = np.sum(deltas[i], axis=0).reshape(self.biases[i].shape)
            self.biases[i] -= db * learning_rate/batch_size

    def train(self, x, y, learning_rate=6, iters=1000, tol=0.001):
        labels = np.zeros((len(y), 2))
        for i in range(len(y)):
            labels[i, y[i]] = 1
        prev_cost = self.cost_function(x, labels)
        print(f"cost before: {prev_cost}")
        for c in range(iters):
            self.train_batch(x, labels, learning_rate)
            cost = self.cost_function(x, labels)
            print(f"cost in {c} iteration : {cost}")
            if abs(cost - prev_cost) < tol:
                break
            prev_cost = cost
        print("After train")
        for i in range(self.layers_count):
            print(f"layer {i} : ")
            print("weights")
            print(self.weights[i])
            print("biases")
            print(self.biases[i])

    def predict(self, x):
        self.forward_prop(x)
        return np.argmax(self.activations[-1], axis=1)


class LogisticRegression:
    def __init__(self, x, y, alpha=0.1):
        self.x = np.append(np.ones(x.shape[0]).reshape(x.shape[0], 1), x ,axis=1)
        self.y = y
        self.theta = np.zeros((self.x.shape[1], 1))
        self.alpha = alpha

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def hypothesis(self):
        return self.sigmoid(self.x.dot(self.theta))

    def cost_function(self):
        return -1 * (np.transpose(self.y).dot(np.log(self.hypothesis())) +
                     np.transpose(1 - self.y).dot(np.log(1 - self.hypothesis())))

    def gradient(self):

        return np.transpose(self.x).dot(self.hypothesis() - self.y)

    def train(self, iters=4):
        for i in range(iters):
            dt = self.alpha * self.gradient()
            print(f"ITER {i}, COST: {self.cost_function()[0][0]:.2f}, dTheta = {self.alpha} * GRAD, "
                  f"\n GRAD =  x * (h(x)- y)\n x: \n{self.x} \n h(x) : \n {self.hypothesis()} \n theta :"
                  f" \n {self.theta} \n y : \n {self.y}  \n dTheta = \n{dt}")
            self.theta = self.theta - dt
        print(f"THETA AFTER TRAIN: \n {self.theta}")

    def plot(self):
        plt.scatter(self.x[:, 1], self.x[:, 2], c=['#ff2200' if z == 0 else '#1f77b4' for z in self.y])
        xm = max(self.x[:, 1]) + 1
        if self.theta[2] != 0:
            plt.plot([0, xm], [(0.5 - self.theta[0]) / self.theta[2],
                               (0.5 - self.theta[0] - self.theta[1] * xm) / self.theta[2]])
        plt.show()


class PSO:
    def __init__(self, x, y, n=10, xmin=-10, xmax=10, ymin=-10, ymax=10, prange=5, c0=0.8, c1=0.1, c2=0.1):
        self.x = np.append(np.ones(x.shape[0]).reshape(x.shape[0], 1), x, axis=1)
        self.y = y
        self.n_particles = n
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.prange = prange
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2

        self.particles = np.random.rand(self.n_particles, 2) * self.prange * 2 - self.prange
        self.v_o = np.random.rand(self.n_particles, 2)
        self.v_n = np.zeros((self.n_particles, 2))
        self.best_particle = copy.copy(self.particles)
        self.best_global = self.particles[0]

    @staticmethod
    def reg_cost(x, y, w):
        v = x.dot(w) - y
        return np.transpose(v).dot(v) / (2 * x.shape[0])

    def do_iter(self):
        for p in range(1, self.particles.shape[0]):
            cost = self.reg_cost(self.x, self.y, self.particles[p, :].reshape(2, 1))
            # update best local score
            if cost < self.reg_cost(self.x, self.y, self.best_particle[p, :].reshape(2, 1)):
                self.best_particle[p, :] = self.particles[p, :]
            # update best global score
            if cost < self.reg_cost(self.x, self.y, self.best_global.reshape(2, 1)):
                self.best_global = self.particles[p]

        for i in range(self.particles.shape[0]):
            print(f"PARTICLE {i} : ")
            r1 = np.random.rand(2)
            r2 = np.random.rand(2)
            self.v_n[i] = self.c0 * self.v_o[i] + self.c1 * r1 * (self.best_particle[i] - self.particles[i]) + \
                     self.c2 * r2 * (self.best_global - self.particles[i])
            print(f"Vn = {self.c0} * Vo + {self.c1} * diag[{r1[0]}, {r1[1]}] * (local_best - actual_position) +"
                  f" {self.c2} * diag[{r2[0]}, {r2[1]}] * (global_best - actual_position) "
                  f"\n Vo : {self.v_o[i][0]},  {self.v_o[i][1]} "
                  f"\n actual position : {self.particles[i][0]}, {self.particles[i][1]} -"
                  f" score : {self.reg_cost(self.x, self.y, self.particles[i].reshape(2, 1))}"
                  f"\n local best : {self.best_particle[i][0]}, {self.best_particle[i][1]} -"
                  f" score : {self.reg_cost(self.x, self.y, self.best_particle[i].reshape(2, 1))}"
                  f"\n global : {self.best_global[0]}, {self.best_global[1]} -"
                  f" score : {self.reg_cost(self.x, self.y, self.best_global.reshape(2, 1))} \n")
        self.v_o = copy.copy(self.v_n)
        self.particles = self.particles + self.v_n

    def animate(self, iter=0):
        plt.figure()
        print(f"\nITER {iter}\n")
        self.do_iter()
        xlist = np.linspace(self.xmin, self.xmax, 100)
        ylist = np.linspace(self.ymin, self.ymax, 100)
        X, Y = np.meshgrid(xlist, ylist)
        Z = np.empty((100, 100))
        for i in range(100):
            for j in range(100):
                Z[i, j] = self.reg_cost(self.x, self.y, np.array([X[i, j], Y[i, j]]).reshape(2, 1))
        cp = plt.contourf(X, Y, Z)
        plt.colorbar(cp)

        for i in range(self.n_particles):
            plt.quiver(self.particles[i, 0], self.particles[i, 1], self.v_o[i, 0], self.v_o[i, 1], color="red", width=0.005)
        plt.scatter(self.particles[:, 0], self.particles[:, 1], c="#000000")
        plt.title(f"Iter: {iter}, best value : {self.reg_cost(self.x, self.y, self.best_global.reshape(2, 1))[0][0]:.2f} at {self.best_global}")
        plt.show()

    def plot_reg(self):
        plt.figure()
        plt.title("fitted regression")
        xmin = min(self.x[:, 1]) - 1
        xmax = max(self.x[:, 1]) + 1
        plt.plot([xmin, xmax], [self.best_global[0], self.best_global[0] + xmax * self.best_global[1]])
        plt.scatter(self.x[:, 1], self.y)
        plt.show()


def test_PSO():
    p = PSO(np.array([20, 40, 60, 80]).reshape(4, 1), np.array([210, 430, 580, 670]).reshape(4, 1), 4)
    for i in range(10):
        p.animate(i)
    p.plot_reg()

def NN_test():
    df = read_csv("D:/Projects/Data/data_logistic.csv")
    x = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy().reshape(x.shape[0], 1)
    n = NeuralNet(2, [4], 2)
    n.train(x, y, learning_rate=6)
    plt.scatter(x[:, 0], x[:, 1],
                c=['#ff2200' if z[0] == 0 else '#1f77b4' for z in y])
    (x_min, x_max) = np.min(x[:, 0]) - 1, np.max(x[:, 0]) + 1
    (y_min, y_max) = np.min(x[:, 1]) - 1, np.max(x[:, 1]) + 1
    # arbitrary number
    elements = y.shape[0] * 40
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, elements), np.linspace(y_min, y_max, elements))
    p = np.empty((elements, elements))
    for i in range(elements):
        for j in range(elements):
            k = np.array([x_grid[i, j], y_grid[i, j]]).reshape(1, 2)
            p[i, j] = n.predict(k)

    plt.contour(x_grid, y_grid, p, levels=[0.5])
    plt.title("title")
    plt.show()

def test_LR():
    lr = LogisticRegression(np.array([[0, 0], [1, 1], [2, 4], [2, 0], [2, 1], [3, 2]]), np.array([0, 0, 0, 1, 1, 1]).reshape(6, 1))
    lr.train()
    lr.plot()


def main():
    np.random.seed(8)
    print("------------------------------Logistic Regression-----------------------------------------")
    test_LR()

    print("------------------------------Neural Network----------------------------------------------")
    NN_test()

    print("------------------------------PSO---------------------------------------------------------")
    test_PSO()


if __name__ == "__main__":
    main()
