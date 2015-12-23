import numpy as np
import random
from matplotlib import pyplot 
from PIL import Image
import matplotlib.cm as cm

epsilon = 10**-6

#display macros from prev problem
def scale2rgb(data):
	max = np.amax(data)
	min = np.amin(data)
	data = data - min
	data = data/(max - min)
	data = 255*data
	return data.astype('int')

def display (data):
	rgbdata = scale2rgb(data)
	pyplot.imshow(rgbdata.reshape(28,28), cmap=cm.Greys_r)
	pyplot.show()


#power method
def powermethod (A, N):
	d = A.shape[0]
	v = np.asarray([random.random() for i in range(d)])
	l = 1
	oldlambda = 0
	lambdalist = [l]
	i = 0
	while abs(l - oldlambda) >= epsilon:
		if(i==N):
			break
		else:
			oldlambda=l
			v = A.dot(v)
			newlambda = np.linalg.norm(v)
			v = v/newlambda
			l=newlambda
			lambdalist.append(l)
			i = i +1
	return (l,v, lambdalist)

#data/covariance matrix from (4)
lfa = np.genfromtxt("lfa.txt")
def covmatrix(data, N):
	C = np.zeros((784,784))
	mu = np.mean(data, axis=0, keepdims =True)
	for x in data[:N]:
		C = C + ((x-mu).T.dot(x-mu))
	C = C / float(N)
	return C

largeN = 1000000000000
#dominant eigenvector/value for N=50
print("Dominant eigenvector for N=50:")
print(powermethod(covmatrix(lfa, 50), largeN)[1])
print("Dominant eigenvalue for N=50:")
print(powermethod(covmatrix(lfa, 50), largeN)[0])

#dominant eigenvector/value for N=2000
print("Dominant eigenvector for N=2000:")
print(powermethod(covmatrix(lfa, 2000), largeN)[1])
print("Dominant eigenvalue for N=2000:")
print(powermethod(covmatrix(lfa, 2000), largeN)[0])

#plot for N=50
lambdaiterations = powermethod(covmatrix(lfa, 50), largeN)[2]
pyplot.plot(range(1,6), lambdaiterations[:5], '.r-') 
pyplot.show()

#plot for N=2000
lambdaiterations = powermethod(covmatrix(lfa, 2000), largeN)[2]
pyplot.plot(range(1,6), lambdaiterations[:5], '.r-') 
pyplot.show()

#images for N=50
display(powermethod(covmatrix(lfa, 50),0)[1])
display(powermethod(covmatrix(lfa, 50),2)[1])
display(powermethod(covmatrix(lfa, 50),5)[1])


#images for N=2000
display(powermethod(covmatrix(lfa, 2000),0)[1])
display(powermethod(covmatrix(lfa, 2000),2)[1])
display(powermethod(covmatrix(lfa, 2000),5)[1])




