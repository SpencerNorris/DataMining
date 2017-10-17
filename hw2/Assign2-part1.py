#!/usr/bin/env python3
'''
Author: Spencer Norris
Date: 09/26/17
Description: Let's get some principal components.
'''
import matplotlib.pyplot as plt
import numpy as np
import math
import sys



def scatter_plot(D,title):
	plt.figure()
	plt.scatter(D[:,0], D[:,1], color='blue', alpha=0.5)
	plt.title(title)
	plt.xlabel('PC1')
	plt.ylabel('PC2')
	plt.show()


def _center_kernel_matrix(K):
	'''
	Subtracts the average vector mu from each row of D,
	thus centering the data about the origin.
	'''
	#t = identity matrix times the ones matrix divided by n
	t = np.identity(len(K)) - ( np.ones((len(K), len(K))) * (1.0/float(len(K))) )
	return np.matmul(np.matmul(t,K), t)


def _get_norm(x):
	'''
	Returns the L2 norm of the input vector x.
	'''
	return math.sqrt(sum(map(lambda i: i**2, x)))


def _project_b_onto_a(b,a):
	return (np.dot(a,b) / _get_norm(a)**2)


def linear_kernel(A,B,target_var):
	'''
	Accepts two vectors A and B and computes 
	the linear kernel function between them.
	This should theoretically be equivalent
	to computing linear PCA!
	'''
	return np.dot(A,B)


def gaussian_kernel(A,B,target_var):
	'''
	Accepts two column attribute vectors A and B
	and computes the Radial Basis Function kernel (e.g. Gaussian kernel).
	'''
	return math.exp(-1 * (_get_norm(A - B)**2 / (2 * target_var)) )


def pca_with_covar_matrix(D, target_var):
	'''
	Computes the principal components of the matrix
	using standard PCA, which is theoretically equivalent
	to linear PCA.
	'''

	def __get_avg_vector(D):
		'''
		Calculates mu for the input array D
		'''
		return np.mean(D, axis=0)
	def __center_data(D, mu):
		'''
		Subtracts the average vector mu from each row of D,
		thus centering the data about the origin.
		'''
		return D - np.transpose(mu[:, None])

	def _get_covariance_matrix(Z):
		'''
		Computes the covariance matrix using the dot product.
		'''
		return np.array([ 
					[np.dot(Z[:,i], Z[:,j]) / len(Z) for i in range(len(Z.T))] 
					for j in range(len(Z.T))
				])

	def __project_b_onto_a(b,a):
		return (np.dot(a,b) / __get_norm(a)**2)


	print("########COVARIANCE PCA########")

	mu = __get_avg_vector(D)
	Z = __center_data(D, mu)
	sigma = _get_covariance_matrix(Z)

	#Get eigenvalues, eigenvectors of covariance matrix
	eig_vals, eig_vecs = np.linalg.eigh(sigma)

	#Flip eig_vals, eig_vecs
	eig_vecs = np.fliplr(eig_vecs)
	eig_vals = eig_vals[::-1]

	#Filter out eigvals, vecs where eigval <= 0
	selections = np.array(list(  map(lambda eig: True if eig > 0 else False, eig_vals)  ))
	eig_vecs = eig_vecs[:, selections]
	eig_vals = list(filter(lambda x: x > 0, eig_vals))

	print("Eigenvalues: \n", eig_vals)
	print("Eigenvectors: \n", eig_vecs)

	#project data onto first two principal components
	C_r = eig_vecs[:,:2]
	return np.matmul(Z, C_r)


def _kernel_pca(D, kernel_f, target_var):
	'''
	Performs kernel principal component analysis, of course!
	'''

	print("######## KERNEL PCA ########")

	#Populate kernel matrix
	n = len(D)
	K = np.zeros((n,n))
	for i in range(len(D)):
		for j in range(len(D)):
			K[i,j] = kernel_f(D[i], D[j], target_var)
	print("Kernel Matrix: \n", K)
	print(K.shape)

	#Center kernel data
	K = _center_kernel_matrix(K)	

	#Get eigenvalues, eigenvectors
	eig_vals, eig_vecs = np.linalg.eigh(K)

	#Flip eig_vals, eig_vecs
	eig_vecs = np.fliplr(eig_vecs)
	eig_vals = eig_vals[::-1]

	#Filter out eigvals, vecs where eigval <= 0
	selections = np.array(list(  map(lambda eig: True if eig > 0 else False, eig_vals)  ))
	eig_vecs = eig_vecs[:, selections]
	eig_vals = list(filter(lambda x: x > 0, eig_vals))
	assert(len(eig_vals) == len(eig_vecs.T))

	#Compute variances
	variances = [( lambda x: float(eig) / float(len(eig_vals)) )(eig) for eig in eig_vals]

	#Normalize eigenvectors
	eig_vecs = np.array([math.sqrt(1/eig_vals[i]) * eig_vecs.T[i] for i in range(len(eig_vals))]).T
	print("Eigenvalues: \n", eig_vals[:len(D.T)])
	print("Normalized Eigenvectors: \n", eig_vecs[:,:len(D.T)])

	#Get eigvecs up to and including total variance
	dim = 0
	total_var = 0.0
	while(total_var / sum(variances) < target_var):
		try:
			total_var += variances[dim]
			dim += 1
		except IndexError:
			#Walked over edge, can't capture that much variance w/ this KPCA!
			dim = len(K)
			break

	if(dim >= len(D.T)):
		print("Couldn't reduce dimensionality for variance ", target_var, "!")
	print("Reduced Num. Dimensions: ", dim)

	#Get reduced dim. data for d=2
	C_r = eig_vecs[:,:2]
	A = np.array([np.matmul(C_r.T, K[i]) for i in range(len(K))])
	return A


def main(f,target_var):
	D = np.genfromtxt(f, delimiter=',')

	#Perform kernel PCA using the linear kernel, plot data
	A = _kernel_pca(D, linear_kernel, target_var)
	scatter_plot(A, "Linear Kernel PCA")

	#Perform standard PCA using a covariance matrix
	B = pca_with_covar_matrix(D, target_var)
	scatter_plot(B, "Standard PCA")

	#Perform Gaussian kernel PCA 
	C = _kernel_pca(D, gaussian_kernel, target_var)
	scatter_plot(C, "Gaussian Kernel PCA")

	return 0


if __name__ == '__main__':
	f_name = sys.argv[1]
	spread = float(sys.argv[2])
	sys.exit(main(f_name, spread))