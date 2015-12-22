import numpy as np
import random
from matplotlib import pyplot 

#(2)(a) Plot toy_data with k=4 
#in which each point's color indicates its cluster

def kmeans(dataSet, k):
	# Initialize k random centroids
	rindex =  np.array(random.sample(range(len(dataSet)), k)) 
	centroids = dataSet[rindex,:]

	oldcentroids = [] 

	while not np.array_equal(oldcentroids, centroids):
		oldcentroids = centroids
		
		#Update cluster assigments
		#clusters = dictionary where clusters[i] = points in centroids[i]'s cluster
		clusters = {} 
		for point in dataSet:
			centroidindex = min([(i, np.linalg.norm(point - centroids[i])) for i in range(k)],
			key=lambda t:t[1])[0]
			if centroidindex in clusters:
				clusters[centroidindex].append(point)
			else:
				clusters[centroidindex] = [point]

		#Update centroids
		newcentroids = []
		for centroidindex in clusters:
			if len(clusters[centroidindex])> 0:
				newcentroid = np.mean(clusters[centroidindex], axis = 0)
				newcentroids.append(newcentroid)
			else: 
				return print("error cluster empty, rerun program")
		centroids = newcentroids

	return (centroids, clusters)

toydata = np.genfromtxt("toy_data.txt")

def plotcluster(cluster, c):
	X = cluster[:,0]
	Y = cluster[:,1]
	return pyplot.scatter(X,Y,color= c)

kmeansclusters = kmeans(toydata, 4)[1]
array0 = np.asarray(kmeansclusters[0])
array1 = np.asarray(kmeansclusters[1])
array2 = np.asarray(kmeansclusters[2])
array3 = np.asarray(kmeansclusters[3])
plotcluster(array0, 'red')
plotcluster(array1, 'blue')
plotcluster(array2, 'green')
plotcluster(array3, 'purple')
pyplot.show()
