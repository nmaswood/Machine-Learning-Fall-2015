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

def display (data):
	rgbdata = scale2rgb(data)
	plt.imshow(rgbdata.reshape(28,28), cmap=cm.Greys_r)
	plt.show()

#random image	
image0 = lfa[0,:]
display(image0)

#mean image
mu = np.mean(lfa, axis=0)
display(mu)