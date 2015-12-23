import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

lfa = np.genfromtxt("lfa.txt")

def scale2rgb(data):
	max = np.amax(data)
	min = np.amin(data)
	data = data - min
	data = data/(max - min)
	data = 255*data
	return data.astype('int')

def covmatrix(data, N):
	C = np.zeros((784,784))
	mu = np.mean(data, axis=0, keepdims =True)
	for x in data[:N]:
		C = C + ((x-mu).T.dot(x-mu))
	C = C / float(N)
	return C

def display(data, N):
	cov = covmatrix(data, N)
	evalues, evectors = np.linalg.eig(cov)
	evectors = np.transpose(evectors)
	rgb_evectors = scale2rgb(evectors)
	for i in range(10):
		plt.imshow(rgb_evectors[i].reshape(28,28), cmap=cm.Greys_r)
		plt.show()
#N=50
display(lfa, 50)

#N=100
display(lfa, 100)

#N=1000
display(lfa, 1000)

#N=2000
display(lfa, 2000)


