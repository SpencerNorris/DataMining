#!/usr/bin/env python3
'''
Author: Spencer Norris
Date: 09/26/17
Description: Let's get some angles.
'''
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import sys



def main(d):

	def __get_norm(x):
		'''
		Returns the L2 norm of the input vector x.
		'''
		return math.sqrt(sum(map(lambda i: i**2, x)))

	def __get_vector_cos(a, b):
		'''
		Returns the value for the cosine of the angle between vectors a and b.
		'''
		mag_a = __get_norm(a)
		mag_b = __get_norm(b)
		return np.dot(a, b) / (mag_a * mag_b)

	def __flip_coin():
		return 1 if random.random() >= .5 else -1



	#Randomly generate 100,000 half-diagonals in d-dimensional space
	n = 100000
	half_diags = []
	for i in range(n):
		half_diags.append( (
							np.array([__flip_coin() for x in range(d)]),
							np.array([__flip_coin() for x in range(d)])
					) )

	#Compute angle in degrees for each pair
	angles = []
	for hd in half_diags:
		cos = __get_vector_cos(hd[0], hd[1])
		angle = np.arccos(cos)
		angles.append(np.degrees(angle))

	#Max, min, range, mean and variance
	angles_min = min(angles)
	angles_max = max(angles)
	angles_mean = float(sum(angles)) / float(len(angles))
	print("Max: ", angles_max)
	print("Min: ", angles_min)
	print("Range: ", abs(angles_max - angles_min))
	print("Mean: ", angles_mean)
	print("Variance: ", sum([math.pow((angle - angles_mean), 2) for angle in angles]) / len(angles))


	#Compute Empirical Probability Mass Function (EPMF)
	val_counts = {}
	for angle in angles:
		try:
			val_counts[angle] += 1
		except KeyError:
			val_counts[angle] = 1

	for k in val_counts.keys():
		val_counts[k] = float(val_counts[k]) / float(len(angles))

	#Plot EPMF
	plt.figure()
	plt.bar(list(val_counts.keys()), list(val_counts.values()), 10, color='b')
	plt.title("Empirical Probability Mass Function (EPMF) of Angles of Half-Diagonals on Hypercube of d=%d" % d)
	plt.show()

	return 0


if __name__ == '__main__':
	'''
	Run $ python Assign2-part2.py <YOUR VALUE FOR D>
	'''
	d = int(sys.argv[1])
	sys.exit(main(d))