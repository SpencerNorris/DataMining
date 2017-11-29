#!/usr/bin/env python3

from random import shuffle
from copy import copy
import numpy as np
import sys


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


def __get_error(kernel_f, data, alpha, index, spread):
	true = data[index, len(data.T) - 1]
	h = sum([
		alpha[i] * data[i,len(data.T) - 1] * kernel_f(data[i], data[index], spread)
	for i in range(len(alpha))])
	return h - true


def main(training_data_path, test_data_path, C, eps, kernel_f, spread):
	#Load datasets
	training_data = np.genfromtxt(training_data_path, delimiter=',')
	test_data = np.genfromtxt(test_data_path, delimiter=',')

	#Perform Sequential Minimum Optimization
	tolerance = 0.00001
	alpha = np.array([0 for i in range(len(training_data))])
	tryall = True
	while True:
		alpha_prev = copy(alpha)
		for j in range(len(training_data)):

			#Tryall, tolerance cases
			if tryall == False and (alpha[j] - tolerance < 0 or alpha[j] + tolerance > C):
				continue

			#Iterate on i
			I = list((range(len(training_data))))
			shuffle(I)
			for i in I:
				if i == j:
					continue

				#Tryall, tolerance cases
				if tryall == False and (alpha[i] - tolerance < 0 or alpha[i] + tolerance > C):
					continue

				#Extract x,y; compute k_ij
				x_i = training_data[i, :len(training_data.T) - 1]
				y_i = training_data[i, len(training_data.T) - 1]
				x_j = training_data[j, :len(training_data.T) - 1]
				y_j = training_data[j, len(training_data.T) - 1]
				k = kernel_f(x_i, x_i, spread) + kernel_f(x_j, x_j, spread) - 2*kernel_f(x_i, x_j, spread)
				if k == 0:
					continue

				alpha_i_prime = alpha[i]
				alpha_j_prime = alpha[j]

				#Compute L, H
				if not y_i == y_j:
					L = max(0, alpha_j_prime - alpha_i_prime)
					H = min(C, C - alpha_i_prime + alpha_j_prime)
				if y_i == y_j:
					L = max(0, alpha_i_prime + alpha_j_prime - C)
					H = min(C, alpha_i_prime + alpha_j_prime)
				if L == H:
					continue

				#Compute errors
				E_i = __get_error(kernel_f,training_data, alpha, i, spread)
				E_j = __get_error(kernel_f,training_data, alpha, j, spread)
				print("Error i: ", E_i)
				print("Error j: ", E_j)
				print("L: ", L)
				print("H: ", H)
				print("alpha[j]: ", alpha_j_prime + ( (y_j * (E_i - E_j)) / k ))

				#Adjust alpha values
				alpha[j] = -1 * (alpha_j_prime + ( (y_j * (E_i - E_j)) / k ))
				if alpha[j] < float(L):
					alpha[j] = L
				elif alpha[j] > float(H):
					alpha[j] = H
				alpha[i] = alpha_i_prime + y_j*y_i*(alpha_j_prime - alpha[j])
				print("alpha[j], again: ", alpha[j])
				print("alpha[i]: ", alpha[i])

		if tryall:
			tryall = False
		if(np.linalg.norm(alpha - alpha_prev)) <= eps:
			break


	print("Alpha: ", alpha)
	return 0


if __name__ == '__main__':
	training_data_path = sys.argv[1]
	test_data_path = sys.argv[2]
	C = float(sys.argv[3])
	eps = float(sys.argv[4])
	kernel_arg = sys.argv[5]
	kernel_f = {
		'linear' : linear_kernel,
		'quadratic' : quadratic_kernel,
		'gaussian' : gaussian_kernel
	}[kernel_arg]
	spread = None
	if kernel_arg == 'gaussian':
		spread = float(sys.argv[6])
	sys.exit(main(training_data_path, test_data_path, C, eps, kernel_f, spread))