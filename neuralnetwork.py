import numpy as np
import random
import pickle

class neural_network:

    def __init__(self, num_layers, num_nodes, activation_function, cost_function):
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.layers = []
        self.cost_function = cost_function

        if not num_layers == len(num_nodes):
            raise ValueError("Number of layers must match number node counts")

        for i in range(num_layers):
            if i != num_layers-1:
                layer_i = layer(num_nodes[i], num_nodes[i+1], activation_function[i])
            else:
                layer_i = layer(num_nodes[i], 0, activation_function[i])
            self.layers.append(layer_i)

    def check_training_data(self, batch_size, inputs, labels):
        self.batch_size = batch_size
        if not len(inputs) % self.batch_size == 0:
            raise ValueError("Batch size must be multiple of number of inputs")
        if not len(inputs) == len(labels):
            raise ValueError("Number of inputs must match number of labels")
        for i in range(len(inputs)):
            if not len(inputs[i]) == self.num_nodes[0]:
                raise ValueError("Length of each input data must match number of input nodes")
            if not len(labels[i]) == self.num_nodes[-1]:
                raise ValueError("Length of each label data must match number of output nodes")

    def train(self, batch_size, inputs, labels, validation_input, validation_targets, num_epochs, learning_rate):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # self.check_training_data(self.batch_size, inputs, labels)

        loss_list = []
        accuracy_list = []
        accuracy = 0

        for j in range(num_epochs):
            self.error = 0
            i = 0
            print("== EPOCH: ", j+1, "/", num_epochs, " ==")
            while i+batch_size != len(inputs):
                print("Training with ", i+batch_size+1, "/", len(inputs), end="\r")
                # self.error = 0
                self.forward_pass(inputs[i:i+batch_size])
                self.calculate_error(labels[i:i+batch_size])
                self.back_pass(labels[i:i+batch_size])
                i += batch_size
            self.error /= batch_size
            print("\nError: ", self.error)
            loss_list.append(self.error)
            accuracy = self.validation(validation_input, validation_targets)
            accuracy_list.append(accuracy)

        with open('wide_loss_list.pickle', 'wb') as wf:
            pickle.dump(loss_list, wf)

        with open('wide_accuracy_list.pickle', 'wb') as wf:
            pickle.dump(accuracy_list, wf)


    def forward_pass(self, inputs):
        self.layers[0].activations = inputs
        for i in range(self.num_layers-1):
            temp = np.add(np.matmul(self.layers[i].activations, self.layers[i].weights_for_layer), self.layers[i].bias_for_layer)
            if self.layers[i+1].activation_function == "softmax":
                self.layers[i+1].activations = self.softmax(temp)
            elif self.layers[i+1].activation_function == "relu":
                self.layers[i+1].activations = self.relu(temp)
            else:
                self.layers[i+1].activations = temp

    def relu(self, layer):
        layer[layer < 0] = 0
        return layer

    def relu_prime(self, layer):
        layer[layer < 0] = 0
        layer[layer > 0] = 1
        return layer

    def softmax(self, layer):
        softmax_result = np.empty((0,10))

        for data in layer:
            exp = np.exp(data - np.max(data))
            # softmax_tmp =  exp/np.sum(exp)
            softmax_tmp = exp/ exp.sum(axis=0)
            softmax_result = np.append(softmax_result, [softmax_tmp], axis=0)

        # print("-------- softmax-result ---------")
        # print(softmax_result[0:10])
        return softmax_result

    def calculate_error(self, labels):
        if self.cost_function == "mean_squared":
            self.error += np.mean(np.divide(np.square(np.subtract(labels, self.layers[self.num_layers-1].activations)), 2))

        elif self.cost_function == "cross_entropy":
            # print("----- loss ----")
            # print(np.negative(np.sum(np.multiply(labels, np.log(self.layers[self.num_layers-1].activations)))))
            result = self.layers[self.num_layers-1].activations
            result[result == 0] = 0.000001
            # self.error += np.negative(np.sum(np.multiply(labels, np.log(result))))
            # loglikelihood = -np.log(result)
            loss = np.sum(np.multiply(labels, -np.log(result))) / len(labels)
            self.error = loss
            # self.layers[self.num_layers-1].activations

    def back_pass(self, labels):
        # if self.cost_function == "cross_entropy" and self.layers[self.num_layers-1].activation_function == "softmax":
        targets = labels
        i = self.num_layers-1
        y = self.layers[i].activations    #softmax_layer_activation

        # deltab_1 = np.multiply(y, np.multiply(1-y, targets-y))
        error = y - targets
        deltab = error
        deltaw = np.matmul(np.asarray(self.relu_prime(self.layers[i-1].activations)).transpose(), deltab)

        new_weights = self.layers[i-1].weights_for_layer - self.learning_rate * deltaw
        new_bias = self.layers[i-1].bias_for_layer - self.learning_rate * deltab

        for i in range(i-1, 0, -1):
            y = self.relu_prime(self.layers[i].activations)
            # print(y)
            # y = self.layers[i].activations
            deltab = np.multiply(y, np.multiply(1-y, np.sum(np.multiply(new_bias, self.layers[i].bias_for_layer)).T))
            deltaw = np.matmul(np.asarray(self.layers[i-1].activations).T, np.multiply(y, np.multiply(1-y, np.sum(np.multiply(new_weights, self.layers[i].weights_for_layer),axis=1).T)))
            self.layers[i].weights_for_layer = new_weights
            self.layers[i].bias_for_layer = new_bias

            new_weights = self.layers[i-1].weights_for_layer - self.learning_rate * deltaw
            new_bias = self.layers[i-1].bias_for_layer - self.learning_rate * deltab
        self.layers[0].weights_for_layer = new_weights
        self.layers[0].bias_for_layer = new_bias


    def predict(self, input):
        # self.batch_size = len(input)
        print(len(input))
        self.forward_pass(input[0:100])
        a = self.layers[self.num_layers-1].activations

        predict_result = np.empty((0,10))
        for arr in a:
            arr[np.where(arr==np.max(arr))] = 1
            arr[np.where(arr!=np.max(arr))] = 0
            predict_result = np.append(predict_result, [arr], axis=0)

        return predict_result

    def validation(self, validation_input, validation_targets):
        data = np.empty((0,784))
        targets = []

        for val in range(100):
            ind = random.randrange(0,len(validation_input),1)
            data = np.append(data, [validation_input[ind]], axis=0)
            targets.append(validation_targets[ind])

        predict_result = self.predict(data)
        print(targets)

        predict_labels = []
        for arr in predict_result:
            label = np.argmax(arr)
            predict_labels.append(label)
        print(predict_labels)

        accuracy = self.caculate_accuracy(predict_labels, targets)

        return accuracy

    def caculate_accuracy(self, predict_labels, targets):
        total=0
        correct=0
        for i in range(0, len(targets)):
            total += 1
            if np.equal(targets[i], predict_labels[i]).all():
                correct += 1

        accuracy = correct*100/total
        print("Accuracy: ", accuracy)

        return accuracy

class layer:
    def __init__(self, num_nodes_in_layer, num_nodes_in_next_layer, activation_function):
        self.num_nodes_in_layer = num_nodes_in_layer
        self.activation_function = activation_function
        self.activations = np.zeros([num_nodes_in_layer,1])
        if num_nodes_in_next_layer != 0:
            self.weights_for_layer = np.random.normal(0, 0.001, size=(num_nodes_in_layer, num_nodes_in_next_layer))
            self.bias_for_layer = np.random.normal(0, 0.001, size=(1, num_nodes_in_next_layer))
        else:
            self.weights_for_layer = None
            self.bias_for_layer = None
