#!/usr/bin/env python3

import numpy as np
import math
import sys


def sigmoid(x):
	if x == 0:
		return .5
	x = np.clip(-1000,1000,x)
	return 1 / (1 + math.exp(-x))

def ReLU(x):
	return np.clip(0,1000,x) if x > 0 else 0.0

def squared_dist(X,Y):
	return .5 * (np.linalg.norm(X - Y)**2)

class NeuralNet():
	'''
	Class for wrapping basic data structures representing
	a feedforward neural network.
	'''

	def __init__(self, size_x, num_classes, 
				 depth, dims, learning_rate=.01, 
				 activation_fn=sigmoid, error_fn=squared_dist):
		'''
		size_x = num elements in each input element X
		num_classes = number of discrete classes to predict on
		depth = num hidden layers,
		size_hidden_layer = dimensionality of hidden layers
		activation_fn = activation function to apply to output of hidden layers
		dims = size of hidden layers

		self.weights is a dictionary mapping from the layer in question to a matrix
		containing the weights flowing from that layer to the next.
		'''
		self.weights = {}
		self.num_classes = num_classes
		self.depth = depth
		self.activation = activation_fn
		self.dims = dims
		self.__error_fn = error_fn
		self.learning_rate = learning_rate

		#Input-to-hidden weights (Include +1 for the bias terms)
		self.weights[0] = np.random.uniform(-1, 1, (size_x + 1) * dims).reshape(size_x + 1, dims)

		#Hidden-to-hidden weights
		for l in range(1, depth - 1):
			self.weights[l] = np.random.uniform(-1, 1, (dims + 1) * dims).reshape(dims + 1, dims)

		#Hidden-to-output weights
		self.weights[depth] = np.random.uniform(-1, 1, (dims + 1) * num_classes).reshape(dims + 1, num_classes)


	def __output_fn(self, X):
		'''
		Returns the output vector based on the conditional probability
		of the particular class given the observed X.
		'''
		vals = []
		for x in X.T[0]:
			vals.append(sigmoid(x))

		#Discretize
		i = vals.index(max(vals))
		for k in range(self.num_classes):
			if k == i:
				vals[k] = 1
			else:
				vals[k] = 0

		return vals

	def __backprop(self, result, true_val, layer, intermediates):
		'''
		Update weights in network according to error produced during training.
		'''
		#Calculate deltas for output layer
		delta_outs = [
			(true_val[j] - result[j]) * true_val[j] * (1 - result[j])
			for j in range(self.num_classes)
		]

		#Update hidden-to-output weights using delta_outs
		for i in range(len(self.weights[1])):
			for j in range(len(self.weights[1][i])):
				gradient = intermediates[1].T[0][i] * delta_outs[j]
				self.weights[1][i][j] = self.weights[1][i][j] - self.learning_rate * gradient


		#Find deltas for hidden layer
		if self.activation is sigmoid:
			delta_hidden = [
				intermediates[1].T[0][j] * (1 - intermediates[1].T[0][j]) * sum([
						delta_outs[k] * self.weights[1][j][k] 
						for k in range(self.num_classes)
					])
				for j in range(self.dims)
			]
		else: #It's ReLU
			delta_hidden = [
				0 if intermediates[1].T[0][j] <= 0
				else sum([
						delta_outs[k] * self.weights[1][j][k] 
						for k in range(self.num_classes)
					])
				for j in range(self.dims)
			]


		#Update input-to-hidden weights using hidden deltas
		for i in range(len(self.weights[0])):
			for j in range(len(self.weights[0][i])):
				gradient = intermediates[0].T[0][i] * delta_hidden[j]
				self.weights[0][i][j] = self.weights[0][i][j] - self.learning_rate * gradient


	def __feedforward(self, X, layer, intermediates, training=False):
		'''
		Implements actual feedforward behavior
		'''
		#If we've moved through the final hidden layer,
		#apply act func and pass to the output function
		if layer == self.depth:
			if not training:
				return self.__output_fn(np.matmul(self.weights[self.depth].T, X))
			else:
				return self.__output_fn(np.matmul(self.weights[self.depth].T, X)), intermediates

		#Get signals, apply activation, reshape and propogate to next layer
		else:
			signals = np.matmul(self.weights[layer].T, X)
			res = [self.activation(s[0]) for s in signals]
			res = np.array([[r] for r in res])
			intermediates.append( np.concatenate((res, [[1.0]])) )
			return self.__feedforward(
				np.concatenate((res, [[1]])),
				layer + 1, 
				intermediates, 
				training
			)


	def train(self, X, Y):
		'''
		Wrapper function for training the network
		using the feedforward function with backpropogation.
		'''
		X = np.array([[x] for x in X])
		X = np.concatenate((X, [[1]]))
		res, intermediates = self.__feedforward(X, 0, [X], True)

		#Compute error on result
		error = self.__error_fn(np.array(res), np.array(Y))

		#Perform backpropogation
		if error > 0:
			self.__backprop(res, Y, self.depth, intermediates)

	def predict(self, X):
		X = np.array([[x] for x in X])
		X = np.concatenate((X, [[1]]))
		return self.__feedforward(X, 0, [X], False)




def main(dtrain, dtest, dims, eta, epochs, activation):
	#Read in data
	D = np.genfromtxt(dtrain, delimiter=',')
	T = np.genfromtxt(dtest, delimiter=',')

	#Get total num of distinct classes in training
	unique_classes = set()
	for row in D:
		unique_classes.add(row[len(row) - 1])
	num_classes = len(unique_classes)

	#Instantiate neural network
	net = NeuralNet(len(D.T) - 1, num_classes, 1, dims, eta, 
				 activation_fn=sigmoid if activation == 'sigmoid' else ReLU,
				 error_fn=squared_dist
			)

	#Train network for some number of epochs
	for e in range(epochs):
		np.random.shuffle(D)
		for example in D:
			X = example[:len(example) - 1]
			y = example[len(example) - 1]

			#Create discrete vector
			Y = [0 for i in range(num_classes)]
			Y[int(y) - 1] = 1

			#Train!
			net.train(X, Y)
	print(net.weights)

	#Get accuracy on test set
	total = 0.0
	for example in T:
		X = example[:len(example) - 1]
		y = example[len(example) - 1]
		Y = [0 for i in range(num_classes)]
		Y[int(y) - 1] = 1
		res = net.predict(X)
		if res == Y:
			total += 1.0

	print("Test Accuracy: ", total / float(len(T)))
	print("\n")

	return 0

if __name__ == '__main__':
	dtrain = sys.argv[1]
	dtest = sys.argv[2]
	dims = int(sys.argv[3])
	eta = float(sys.argv[4])
	epochs = int(sys.argv[5])
	activation = sys.argv[6]
	print("Hidden layer size: ", dims)
	print("Learning rate: ", eta)
	print("Number of epochs: ", epochs)
	print("Activation function: ", activation)
	sys.exit(main(dtrain, dtest, dims, eta, epochs, activation))