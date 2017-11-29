#!/usr/bin/env python3
'''
Author: Spencer Norris
File: assign3-kernel.py
Description: Implementation of ridge regression with options
for the linear, quadratic and Gaussian kernels.
'''

import numpy as np
import math
import sys

alpha = 0.01
spread = None

def linear_kernel(A,B,spread):
	'''
	Accepts two vectors A and B and computes 
	the linear kernel function between them.
	'''
	return np.dot(A,B)

def quadratic_kernel(A,B,spread):
	'''
	Accepts two vectors A and B and returns the 
	'''
	return np.dot(A,B)**2

def gaussian_kernel(A,B,spread):
	'''
	Accepts two column attribute vectors A and B
	and computes the Radial Basis Function kernel
	(e.g. Gaussian kernel).
	'''
	return math.exp(-1 * (np.linalg.norm(A - B)**2 / (2 * spread)) )



def main(training_f, test_f, kernel_f):

	#Pull in training, test sets
	training = np.genfromtxt(training_f, delimiter=',')
	test = np.genfromtxt(test_f, delimiter=',')

	#Extract Y vector, reform to column
	X_training = training[:,:len(training.T) - 1]
	Y_training = np.array([[y] for y in training[:, len(training.T) - 1]])

	#Build kernel matrix
	global spread
	K = np.zeros((len(X_training),len(X_training)))
	for i in range(len(X_training)):
		for j in range(len(X_training)):
			K[i,j] = kernel_f(X_training[i], X_training[j], spread)


	#Get constants
	constants = np.matmul(np.linalg.inv(K + alpha * np.identity(len(K))), Y_training)

	#Make predictions on test set
	predictions = []
	X_test = test[:,:len(test.T)-1]
	for z in X_test:
		prediction = 0.0
		for i in range(len(X_training)):
			prediction += constants[i][0] * kernel_f(X_training[i], z, spread)
		predictions.append(prediction)


	#Determine accuracy
	Y_test = test[:,len(test.T)-1]
	count = 0.0
	Y = test[:,len(test.T) - 1]
	for i in range(len(Y)):
		if np.sign(Y[i]) == np.sign(predictions[i]):
			count +=1
	accuracy = count / len(Y)
	print("Accuracy: ", accuracy)

	return 0

if __name__ == '__main__':
	training_f = sys.argv[1]
	test_f = sys.argv[2]
	kernel_arg = sys.argv[3]
	spread = float(sys.argv[4])
	kernel_f = {
		'linear' : linear_kernel,
		'quadratic' : quadratic_kernel,
		'gaussian' : gaussian_kernel
	}[kernel_arg]
	sys.exit(main(training_f, test_f, kernel_f))