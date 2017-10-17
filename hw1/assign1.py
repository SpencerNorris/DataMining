#!/usr/bin/env python3
'''
Author: Spencer Norris
File: assign1.py
Desc.: Implementation of solution for HW1 in Data Mining with Prof. Zaki at RPI.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

#Global
epsilon = None

def multiply_matrices(A, B):
	'''
	Returns the results of performing matrix multiplication
	on two numpy arrays A and B.
	'''
	return np.array([[np.dot(a,b) for b in B.T] for a in A])


def __get_avg_vector(D):
	'''
	Calculates mu for the input array D
	'''
	return np.mean(D, axis=0)


def __get_norm(x):
	'''
	Returns the L2 norm of the input vector x.
	'''
	return math.sqrt(sum(map(lambda i: i**2, x)))

def __get_total_variance(Z):
	return sum([__get_norm(z)**2 for z in Z]) / len(Z)


def __center_data(D, mu):
	'''
	Subtracts the average vector mu from each row of D,
	thus centering the data about the origin.
	'''
	return D - np.transpose(mu[:, None])


def __get_inner_product_covariance_matrix(Z):
	'''
	Computes the covariance matrix using the dot product.
	'''
	return np.array([ 
				[np.dot(Z[:,i], Z[:,j]) / len(Z) for i in range(len(Z.T))] 
				for j in range(len(Z.T))
			])


def __get_outer_product_covariance_matrix(Z):
	'''
	Computes the covariance matrix using the outer product.
	'''
	return sum([np.outer(Z[i], Z[i]) for i in range(len(Z))]) / len(Z)


def __get_vector_cos(a, b):
	'''
	Returns the value for the cosine of the angle between vectors a and b.
	'''
	mag_a = __get_norm(a)
	mag_b = __get_norm(b)
	return np.dot(a, b) / (mag_a * mag_b)


def __get_correlation_matrix(Z):
	return np.array([
			[__get_vector_cos(i,j) for j in Z.T]
		for i in Z.T])


def __get_eigenvectors(sigma, d):

	def __orthogonalize(a,b):
		'''
		Ensures that b is orthogonal to a.
		'''
		return b - (np.dot(b,a) / np.dot(a,a)) * a


	def __frobenius_norm(X):
		total = 0.0
		for i in range(len(X)):
			for j in range(len(X.T)):
				total += X[i,j]**2
		return math.sqrt(total)

	global epsilon
	X_0 = np.random.rand(d,2)
	for j in range(len(X_0.T)):
		X_0[:,j] = X_0[:,j] / __get_norm(X_0[:,j])

	#Begin iteration
	while True:
		#Perform iteration
		X_1 = multiply_matrices(sigma, X_0)
		X_1[:, 1] = __orthogonalize(X_1[:,0], X_1[:,1])

		#Normalize columns of X_1 to unit length
		X_1[:, 0] = X_1[:, 0] / __get_norm(X_1[:, 0])
		X_1[:, 1] = X_1[:, 1] / __get_norm(X_1[:, 1])

		#Convergence, or store value for next iteration
		if __frobenius_norm(X_1 - X_0) < epsilon:
			return X_1
		else:
			X_0 = X_1


def __get_eigenvalues(sigma, eig):
	'''
	Uses the Rayleigh Quotient to calculate 
	the estimated eigenvalues of the eigenvectors.
	'''
	lambdas = []
	for i in range(len(eig.T)):
		u = np.array(eig.T[0])[np.newaxis]
		sigma_times_u = multiply_matrices(sigma,u.transpose())
		u = u[0]
		lambdas.append(np.dot(sigma_times_u.transpose()[0], u) / np.dot(u,u))
	return lambdas


def __plot_correlation_matrix(rho,data):
	#Heat map
	cax = plt.matshow(rho)
	plt.colorbar(cax)
	plt.show()
	plt.close()

	#Scatter plots
	most_correlated = (None, None, float('-inf'))
	least_correlated = (None, None, float('inf'))
	most_anti_correlated = (None, None, float('inf'))
	for i in range(0,6):
		for j in range(0,6):
			if rho[i,j] > most_correlated[2]:
				most_correlated = (i,j,rho[i,j])
			if rho[i,j] < most_anti_correlated[2]:
				most_anti_correlated = (i,j,rho[i,j])
			if abs(rho[i,j]) < abs(least_correlated[2]):
				least_correlated = (i,j, rho[i,j])


	for dataset in [(most_correlated, "Most Correlated"), 
					(least_correlated, "Least Correlated"), 
					(most_anti_correlated, "Most Anti-Correlated")]:
		i = dataset[0][0]
		j = dataset[0][1]
		plt.title(dataset[1])
		plt.scatter(data[:,i], data[:,j])
		plt.show()
		plt.close()


def __plot_eigenvector_projections(eig, Z):
	'''
	Projects centered data onto eigenvectors of covariance matrix
	and plots points in 2-D.
	'''
	def __project_b_onto_a(b,a):
		return (np.dot(a,b) / __get_norm(a)**2)

	x = []
	y = []
	for z in Z:
		zin = np.array([z])
		tmp = multiply_matrices(eig.T,zin.T)
		x.append(tmp[0])
		y.append(tmp[1])

	plt.scatter(x,y)
	plt.show()


def main(f):
	data = np.genfromtxt(f)

	#Center data, get average vector and total variance
	mu = __get_avg_vector(data)
	print("Avg. Vector (mu): ", mu)
	Z = __center_data(data, mu)
	print("Centered Data: \n", Z)
	print("Total Variance: ", __get_total_variance(Z))

	#Compute covariance matrix
	sigma = __get_inner_product_covariance_matrix(Z)
	assert(sigma.all() == __get_outer_product_covariance_matrix(Z).all())
	print("Covariance Matrix: \n", sigma)

	#Compute correlation
	rho = __get_correlation_matrix(Z)
	print("Correlation Matrix: \n", rho)

	#Compute eigenvectors
	eig = __get_eigenvectors(sigma, len(Z.T))
	print("Eigenvectors: \n",eig)
	print("Eigenvalues: \n", __get_eigenvalues(sigma,eig))

	#Plot results
	__plot_correlation_matrix(rho,data)
	__plot_eigenvector_projections(eig,data)

	return 0


if __name__ == '__main__':
	f = str(sys.argv[1])
	epsilon = float(sys.argv[2])
	sys.exit(main(f))