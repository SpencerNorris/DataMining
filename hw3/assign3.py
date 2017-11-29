#!/usr/bin/env python3
'''
Author: Spencer Norris
File: assign3.py
Description: implementation of logistic regression using stochastic gradient descent.
'''

from copy import copy
import numpy as np
import random
import math
import sys

def __logistic(z):
	'''
	Just the logistic function computed on z.
	'''
	return 1 / (1 + math.exp(-1 * z))


def __get_weights(training, epsilon, eta):
	'''
	Calculates the weights for logistic regression 
	using Stochastic Gradient Descent (SGD).
	'''

	def __compute_gradient(x, y, weights):
		inner = (-1) * y * np.dot(weights,x)
		return y * __logistic(inner) * x

	#Randomly initialize weight vector
	weights = np.array([random.uniform(0,1) for i in range(len(training.T))])

	#Perform gradient descent
	while(True):
		#Make copy of weights
		weights_prev = copy(weights)
		
		#Randomize order of dataset
		np.random.shuffle(training)

		#Extract X, dependent variable Y (and transform Y into column)
		X = training[:,:len(training.T) - 1]
		Y = training[:,len(training.T) - 1]
		Y = np.array([[y] for y in Y])
	

		#Augment X with column of 1's
		ones = np.array([[1] for i in range(len(X))])
		X = np.concatenate((X, ones), axis=1)

		#Iterate over random-order dataset
		for i in range(len(training)):
			x = X[i]
			y = Y[i]
			weights = weights + eta * __compute_gradient(x, y, weights)

		#Check for convergence
		if (np.linalg.norm(weights - weights_prev) < epsilon):
			break

	return weights


def main(training_f, test_f, epsilon, eta):

	#Pull in training, test datasets
	#Note: last column of each dataset = dependent variable
	training = np.genfromtxt(training_f, delimiter=',')
	test = np.genfromtxt(test_f, delimiter=',')

	#Compute weights using Stochastic Gradient Descent
	weights = __get_weights(training, epsilon, eta)
	print("Weights\n----------\n", weights, "\n")

	#Predict classes for each x
	predictions = []
	X = test[:,:len(test.T) - 1]
	ones = np.array([[1] for i in range(len(X))])
	X = np.concatenate((X, ones), axis=1)

	for x in X:
		prediction = __logistic(np.dot(weights,x))
		if prediction >= .5:
			predictions.append(1.0)
		else:
			predictions.append(-1.0)


	#Test accuracy of predictions
	count = 0.0
	Y = test[:,len(test.T) - 1]
	for i in range(len(Y)):
		if Y[i] == predictions[i]:
			count +=1
	accuracy = count / len(Y)
	print("Accuracy: ", accuracy)
	return 0


if __name__ == '__main__':
	training_f = sys.argv[1]
	test_f = sys.argv[2]
	epsilon = float(sys.argv[3])
	eta = float(sys.argv[4])
	sys.exit(main(training_f, test_f, epsilon, eta))