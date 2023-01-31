from nn import *
import time
tools = Tools
np.random.seed(69)
X = [0, 0.5, 1]
y = [0, 1, 0]
X_NN = np.reshape(np.array(X), (len(X), 1))

biases_NN = [[0, 0], [0]]

network = NN(1, 1, 0.1)
network.setup_layers(hlayer_neurons=(2,), olayer_neurons=1, n_inputs=1, inputs=X_NN, n_neurons=3)
network.set_weights(([[2.74, -1.13]], [[0.36], [0.63]]))
network.set_biases(biases_NN)
network.train(y, 100)
print(network.get_biases())

points = [[], []]
n_points = 1000
for p in range(n_points):
    points[0].append(p/n_points)
    points[1].append(network.output(p/n_points)[0])
tools.plot_data(points[0], points[1], 20)

#tools.plot_data(points[0][0], points[1])
