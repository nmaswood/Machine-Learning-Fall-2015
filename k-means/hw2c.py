import numpy as np
import random
from matplotlib import pyplot 
from PIL import Image

#2C, compress bird image using 16 clusters
#SLOW(~5 min) BUT FUNCTIONAL -sorry! 

def kmeans_imagecompress(dataSet, k):
	# Initialize k random centroids
	rindex =  np.array(random.sample(range(128), k)) 
	centroids = dataSet[rindex, rindex, :]

	oldcentroids = [] 

	while not np.array_equal(oldcentroids, centroids):
		oldcentroids = centroids
		
		#Update cluster assigments
		#clusters = dictionary where clusters[i] = points in centroids[i]'s cluster
		clusters = {} 
		
		for l in range(128):
			for w in range(128):
				centroidindex = min([(i, np.linalg.norm(A[l,w,:] - centroids[i])) for i in range(len(centroids))],
				key=lambda t:t[1])[0]
				if centroidindex in clusters:
					clusters[centroidindex].append(A[l,w,:])
				else:
					clusters[centroidindex] = [A[l,w,:]]

		#Update centroids
		newcentroids = []
		for centroidindex in clusters:
			newcentroid = np.mean(clusters[centroidindex], axis = 0)
			newcentroids.append(newcentroid)
		centroids = newcentroids

	return (centroids, clusters)

in_image = Image.open("bird_small.tiff") 
A = np.asarray(in_image)

centroids = kmeans_imagecompress(A, 16)[0]
Acopy = np.zeros( (128,128,3), dtype=np.uint8)

for l in range(128):
	for w in range(128):
		centroidindex = min([(i, np.linalg.norm(A[l,w,:] - centroids[i])) for i in range(len(centroids))],
				key=lambda t:t[1])[0]
		Acopy[l,w,:] = centroids[centroidindex]

out_image = Image.fromarray(Acopy, "RGB") 
out_image.save("output-bird.tiff")



































