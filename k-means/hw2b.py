import numpy as np
import random
from matplotlib import pyplot 

#(2)(b) Plot the value of the distortion fn
#as a function of 20 runs of kmeans

toydata = np.genfromtxt("toy_data.txt")

def kmeans_iterations(dataSet, k, iterations):
	# Initialize k random centroids
	rindex =  np.array(random.sample(range(len(dataSet)), k)) 
	centroids = dataSet[rindex,:]

	oldcentroids = [] 
	distvalues = []
	for i in range(iterations):
		oldcentroids = centroids
		
		#Update cluster assigments
		#clusters = dictionary where clusters[i] = points in centroids[i]'s cluster
		clusters = {} 
		d = 0
		for point in dataSet:
			tuple = min([(i, np.linalg.norm(point - centroids[i])) for i in range(k)],
			key=lambda t:t[1])
			centroidindex = tuple[0]
			d = d+ (tuple[1]**2)
			if centroidindex in clusters:
				clusters[centroidindex].append(point)
			else:
				clusters[centroidindex] = [point]

		#keep track of distortion values
		distvalues.append(d)

		#Update centroids
		newcentroids = []
		for centroidindex in clusters:
			newcentroid = np.mean(clusters[centroidindex], axis = 0)
			newcentroids.append(newcentroid)
		centroids = newcentroids

	return (centroids, clusters, distvalues)

d_iterations = kmeans_iterations(toydata, 4,20)[2]

pyplot.plot(range(1,21), d_iterations, '.r-') 
pyplot.show()