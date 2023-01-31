import numpy as np
import matplotlib.pyplot as plt
np.random.seed(69)
#softplus = log(1+e^x)
class Tools():
    def softplus(x):
        return np.log(1+np.exp(x))

    def plot_data(X, Y, size):
        plt.plot(X, Y, linestyle='None', marker=".", markersize=size)
        plt.show()

class Layer(Tools):
    def __init__(self, n_neurons, n_inputs):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = np.random.randn(n_inputs, n_neurons)
        #self.weights = np.array([[3.34, -3.53]])
        self.biases = np.zeros((1, n_neurons))
        #self.biases = np.array([[-1.43, 0.57]])
        
        self.outputs = []
        self.neuron_outputs = []

    def set_weights(self, weights):
        self.weights = weights

    def set_biases(self, biases):
        self.biases = biases
    
    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

    def get_neurons_outputs(self):
        return self.neuron_outputs


    def forward(self, inputs):
        #print(f"INPUTS --------------------\n{inputs}\n--------------------")
        #print(f"WEIGHTS --------------------\n{self.weights}\n--------------------")
        outputs = []
        #output = np.dot(inputs, self.weights)+biases
        for n in range(self.n_neurons):
            result = np.dot(inputs, self.weights[0][n])+self.biases[0][n]
            self.neuron_outputs.append(result)
            outputs.append(result)
        #print(self.biases)
        #print(f"ZERO --> {self.weights[0]}")
        #print(f"HERE --> {np.dot(inputs, self.weights[0][0])}")
        #print(f"DIMS:\nweight dim --> {self.weights[0][0].shape}\ninputs --> {inputs[0].shape}")
        #print(f"\n\nMath: {np.dot(inputs, self.weights)}")

        output = Tools.softplus(outputs)
        self.outputs = []
        self.outputs.append(output)
        return output


class OutputLayer(Tools):
    def __init__(self, n_neurons, n_inputs):
        self.n_neurons = n_neurons
        self.weights = np.random.randn(n_inputs, n_neurons)
        #self.weights = np.array([[-1.22], [-2.30]])
        self.biases = np.zeros((1, n_neurons))

    def set_weights(self, weights):
        self.weights = weights

    def set_biases(self, biases):
        self.biases = biases
    
    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

    
    def forward(self, inputs):
        #print(f"\n\nINPUTS:\n{inputs}\n")
        #print(f"Neuron outputs sum:\n{np.sum(inputs, axis=0)}\n")
        #print(f"OUTLAYER weigths:\n{self.weights}")
        outputs = []
        #need to mult by w before summation
        #for n in range(self.n_neurons):
            #outputs.append(np.dot(np.sum(inputs, axis=0), self.weights[n])+self.biases[n])
        #print("\nINPUTS:")
        #for i in inputs:
            #print(i)
        output1 = np.dot(inputs[0], self.weights[0])
        output2 = np.dot(inputs[1], self.weights[1])
        #print(f"\nOUT1:\n{output1}")
        #print(f"\nOUT2:\n{output2}")
        outputs.append(np.add(output1, output2)+self.biases)

        #print("\n\n")
        return outputs


class NN():
    def __init__(self, n_hlayers, n_outputs, learning_rate):
        self.n_hlayers = n_hlayers
        self.n_outputs = n_outputs
        self.n_neurons = 0
        
        self.hlayer_neurons = []

        self.hlayers = []
        self.olayer = None
        self.neurons = []

        self.inputs = []
        self.weights = []
        self.biases = []

        self.layers_output = []
        self.neurons_output = []
        self.outputs = []

        self.learning_rate = learning_rate

    def setup_layers(self, hlayer_neurons: tuple, olayer_neurons, n_inputs, inputs, n_neurons):

        self.inputs = inputs
        self.n_neurons = n_neurons

        self.hlayer_neurons = hlayer_neurons
        #problem here, each layer might receive a different number of inputs, n_inputs should be based on the ammount of neurons in the last layer
        for l in hlayer_neurons:
            self.hlayers.append(Layer(l, n_inputs))

        #problem here, each layer might receive a different number of inputs, n_inputs should be based on the ammount of neurons in the last layer
        self.olayer = OutputLayer(olayer_neurons, 2)
        self.update()

    def set_weights(self, layers_weights):
        for i in range(self.n_hlayers):
            self.hlayers[i].set_weights(layers_weights[i])
        self.olayer.set_weights(layers_weights[-1])
        self.update()

    def set_biases(self, layers_biases):
        for i in range(self.n_hlayers):
            #print(f"//////////////////\n{layers_biases[i]}\n//////////////////")
            #self.hlayers[i].set_biases(np.reshape(layers_biases[i], (1, len(layers_biases[i]))))
            self.hlayers[i].set_biases([np.squeeze(np.asarray(layers_biases[i]))])
        self.olayer.set_biases(layers_biases[-1])
        self.update()

    def get_biases(self):
        self.update()
        return self.biases

    def get_weights(self):
        self.update()
        return self.weights

    def set_inputs(self, X):
        self.inputs = X

    def update(self):
        self.biases = []
        self.weights = []
        for hl in self.hlayers:
            self.biases.append(hl.get_biases())
            self.weights.append(hl.get_weights())
            self.neurons_output = hl.get_neurons_outputs()
        self.biases.append(self.olayer.get_biases())
        self.weights.append(self.olayer.get_weights())


    def forward_pass(self, X=None):
        self.layers_output = []
        if X == None:
            #inputs = np.reshape(np.array(self.inputs), (len(self.inputs), 1))
            last_outputs = self.inputs
            self.layers_output.append(self.inputs)
            #for l in self.layers:
            for l in self.hlayers:
                last_outputs = l.forward(last_outputs)
                self.layers_output.append(last_outputs)

            final_pass = self.olayer.forward(last_outputs)
            self.outputs = final_pass

        else:
            #inputs = np.reshape(np.array(self.inputs), (len(self.inputs), 1))
            last_outputs = X
            #for l in self.layers:
            for l in self.hlayers:
                last_outputs = l.forward(last_outputs)

            final_pass = self.olayer.forward(last_outputs)
            self.outputs = final_pass

        self.update()

        return final_pass


    def gradient_descent(self, expected, var, outputs, multiplier):
        #it's always sum(observedi-predictedi)
        #the lower the learning_rate, higher the precision, slower training
        #the higher the learning_rate, lower precision, faster training 
        #test = np.sum(multiplier*(expected-outputs))
        #print(test)
        step = (np.sum(multiplier*(expected-outputs)))*self.learning_rate
        #print(f"STEP == {step}")
        #print(f"VAR == {var}")
        #doing it this way, the -step will indicate wich side the var point should move to
        # if the derivative*learning_rate is already negative, the point will move to the right by "step"
        #the step is used to move the point by (dy/dx)*learning_rate, let's say the learning_rate = 1/10, then it will move the point by 1/10 of its slope
        output = var-step
        return output


    def backpropagation(self, expected):
        self.biases[0] = np.array(self.biases[0]).tolist()
        expected = np.array(expected)
        #biases = np.array(self.biases)
        #print("\n\nBackpropragating B3")
        #print(f"Quantifiying Loss with SSR:")
        #print(f"sum[(expectedi-observedi)²]")
        #print(f"--> {np.sum(np.square(np.array(expected)-np.array(self.outputs)))}")
        #print("\nEq needed --> d/db3(SSR) --> dSSR/db3")
        #print("dSSR/db3 = -2(sum(obersevedi-predictedi))")
        #print(self.biases)
        #new_w3 = self.gradient_descent(expected, )
        #print(self.weights)
        #new_w3 = self.gradient_descent(expected, self.weights[])
        #self.weights = np.reshape(self.weights, (1, len(self.weights)))
        #self.biases = np.reshape(self.biases, (1, self.n_neurons))
        for l in self.hlayer_neurons:
            layer_index = 1
            #print("----------\n\n", self.layers_output)
            for n in range(l):
                if layer_index == len(self.hlayer_neurons):
                    #self.biases[0][layer_index][n] = self.gradient_descent(expected, self.biases[0][layer_index][n], self.outputs, self.weights[-1][])
                    #print(f"New bias:\n", np.sum(-2*(expected-self.outputs)*0.36*np.exp(np.array(self.neurons_output[n]).T)/(1+np.exp(np.array(self.neurons_output[n]).T))*1))
                    for o in range(self.n_outputs):
                        #self.biases[0][layer_index-1][n] = self.gradient_descent(expected, self.biases[0][layer_index-1][n], self.outputs, -2*self.weights[-1][n][o]*np.exp(np.array(self.neurons_output[n]).T)/(1+np.exp(np.array(self.neurons_output[n]).T))*1)
                        #self.biases[0][0][0] = self.gradient_descent(expected, self.biases[0][layer_index-1][n], self.outputs, -2)
                        #print("NEW", self.gradient_descent(expected, self.biases[0][layer_index-1][n], self.outputs, -2*self.weights[-1][n][o]*np.exp(np.array(self.neurons_output[n]).T)/(1+np.exp(np.array(self.neurons_output[n]).T))*1))
                        #print(f"New bias:\n", np.sum(-2*(expected-self.outputs)*self.weights[-1][n][o]*np.exp(np.array(self.neurons_output[n]).T)/(1+np.exp(np.array(self.neurons_output[n]).T))*1))
                        #tmp = self.biases[0][layer_index-1][n]-np.sum(-2*(expected-self.outputs)*self.weights[-1][n][o]*np.exp(np.array(self.neurons_output[n]).T)/(1+np.exp(np.array(self.neurons_output[n]).T))*1)*0.1
                        #print(tmp)
                        self.biases[0][layer_index-1][n] = self.gradient_descent(expected, self.biases[0][layer_index-1][n], self.outputs, -2*self.weights[-1][n][o]*np.exp(np.array(self.neurons_output[n]).T)/(1+np.exp(np.array(self.neurons_output[n]).T))*1)
                        #print(self.biases[0][layer_index-1][n]-np.sum(-2*(expected-self.outputs)*self.weights[-1][n][o]*np.exp(np.array(self.neurons_output[n]).T)/(1+np.exp(np.array(self.neurons_output[n]).T))*1)*0.1)
                        self.set_biases(self.biases)

                        #print("-------------------",np.sum(-2*(expected-self.outputs)*self.weights[-1][n][o])*1)
                        #print("-------------------",np.sum(-2*(expected-self.outputs)*self.weights[-1][n][o]*(np.exp(self.neurons_output[0])/(1+np.exp(self.neurons_output[0]))))*self.inputs)
                        #outs = [[0],[1.37],[2.74]]
                        #inps = [[0], [0.5], [1]]
                        #print(np.sum(-2*(expected-self.outputs)*0.36*np.exp([[0],[1.37],[2.74]])/(1+np.exp([[0],[1.37],[2.74]]))*inps))
                        #print(np.sum(-2*(expected-self.outputs)*0.36*np.exp(np.array(outs).T)/(1+np.exp(np.array(outs).T))*np.array(inps).T))
                        #WORKS --> print(np.sum(-2*(expected-self.outputs)*self.weights[-1][n][o]*np.exp(np.array(self.neurons_output[0]).T)/(1+np.exp(np.array(self.neurons_output[0]).T))*np.array(self.inputs).T))
                        #print(np.sum(-2*(expected-self.outputs)*self.weights[-1][n][o]*np.exp(np.array(self.neurons_output[n]).T)/(1+np.exp(np.array(self.neurons_output[n]).T))*np.array(self.inputs).T))


                        #print(self.weights[0][layer_index-1][n])
                        self.weights[0][layer_index-1][n] = self.gradient_descent(expected, self.weights[0][layer_index-1][n], self.outputs, -2*self.weights[-1][n][o]*np.exp(np.array(self.neurons_output[n]).T)/(1+np.exp(np.array(self.neurons_output[n]).T))*np.array(self.inputs).T)
                        self.set_weights(self.weights)


                        #print(np.sum(((-2*(expected-self.outputs)*0.36)*(np.exp(outs)/(1+np.exp(outs))*inps))))
                        #print(self.neurons_output[0])
                        pass

                #print(self.biases[layer_index][n])
                #0 means it's the biases from some hidden layer, 1 would be the biases from the output layer
                #print(f"OLD{[n]} BIAS:\n {self.biases[0][layer_index-1][n]}")
                #self.biases[0][layer_index][n] = self.gradient_descent(expected, self.biases[0][layer_index][n], self.outputs, self.weights[][])
                #print(f"NEW BIAS{[n]}:\n {self.biases[0][layer_index-1][n]}")
                #print(f"WEIGHTS:\n{self.weights}")
            #print(l)
            layer_index += 1
            pass
        for o in range(self.n_outputs):
            for i in range(len(self.layers_output[-1])):
                #print(self.weights[-1][i][o])
                #self.weights[-1][i][o] = self.gradient_descent(expected, self.weights[-1][i][o], -2*(self.layers_output[-1][i]))
                #print(self.gradient_descent-2*self.layers_output[-1][i])
                #self.weights[-1][i][o] = self.gradient_descent(expected, self.weights[-1][i][o], (self.layers_output[-1][i]), -2)
                #print(f"OLD{[i]}:\n {self.weights[-1][i][o]}")
                #print(f"layers:\n {self.layers_output[-1][i]}")
                #print(f"layers:\n {(expected-self.outputs)*self.layers_output[-1][i]}")
                #print(f"USSR:\n{np.sum((expected-self.outputs)*self.layers_output[-1][i].T*-2)}")
                #print("TEST:\n", np.sum(-2*(expected-self.outputs)*self.layers_output[-1][i]))
                #print(f"NEW{[i]}:\n {self.gradient_descent(expected, self.weights[-1][i][o], self.outputs, -2*(self.layers_output[-1][i]))}")
                #print(f"NEW{[i]}:\n {self.gradient_descent(expected, self.weights[-1][i][o], self.outputs, -2*(self.layers_output[-1][i].T))}")
                self.weights[-1][i][o] = self.gradient_descent(expected, self.weights[-1][i][o], self.outputs, -2*(self.layers_output[-1][i].T))
                self.set_weights(self.weights)
                pass
            self.biases[-1][o] = self.gradient_descent(expected, self.biases[-1][o], self.outputs, -2)
            self.biases[0][0][0] = -1.43
            self.biases[0][0][1] = 0.57
            #self.biases[-1][0] = 2.61
            self.set_biases(self.biases)
        #new_b3 = self.gradient_descent(expected, self.biases[-1][0], -2)
        #self.biases[-1][0] = new_b3
        #print(self.biases)
        #print(f"B3 RESULT --> {new_b3}")

        #self.set_biases(self.biases)

    def train(self, expected, epoch):
        for i in range(epoch):
            self.forward_pass()
            self.backpropagation(expected)

    def output(self, X):
        return self.forward_pass(X)







"""
    def forward():
        #for each layer
        for i in layers:

            #for each neuron in the layer
            for x in range(i):

                print(f"x-1 = {x-1}")
                y = np.array(X)*weights[x-1][]+biases[x-1]
                tools.softplus

        y1 = np.array(X)*weights[0][0]+biases[0]
        y2 = np.array(X)*weights[1][0]+biases[1]

        yact1w = tools.softplus(y1)*weights[0][1]
        yact2w = tools.softplus(y2)*weights[1][1]

        output = np.add(yact1w, yact2w)+biases[2]
"""
