import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
from math import exp
import matplotlib.cm as cm
import itertools
from matplotlib import colors



def kernel(x_0, x_1, tau_squared=.12):

	diff_squared = (x_0 - x_1) ** 2

	return exp(-diff_squared/ (2 * tau_squared))


def get_samples(num_samples= 20):

	return np.random.rand(1,num_samples)

def interval(steps):
	return [i /100. for i in xrange(0,100,steps)]

def posterior(steps = 10):

	zs = interval(steps)

	f_z_list = []

	f_z_list.append(normal(0, 1))


	for cnt in xrange(1,steps):

		z_cnt = zs[cnt]

		k_x = np.asarray([[kernel(z_cnt, x) for x in zs[:cnt]]])
		K = np.zeros((cnt, cnt)) 
		for i in xrange(cnt):
			for j in xrange(cnt):
				K[i][j] = kernel(zs[i],zs[j])


		f = np.transpose(np.asarray([f_z_list]))

		first_op = np.dot(k_x, np.linalg.inv(K))
		mu = np.dot(first_op, f)
		
		second_op = np.dot(k_x, np.linalg.inv(K))

		var = 1 - np.dot(second_op, np.transpose(k_x))

		f_z_list.append(normal(mu, var ** .5))

	return zs, f_z_list 


def show_graph():
	for _ in xrange(20):
		x,y = posterior()
		plt.plot(x,y)

	plt.show()

data = np.genfromtxt("data.dat")
xs = np.asarray([data[:,0]])
ys = np.asarray([data[:,1]])
count = len(xs[0])

def mu_std(input_x):

	k_x = np.asarray([[kernel(input_x, x) for x in xs[0]]])

	K = np.zeros((count, count)) 

	for i in xrange(count):
		for j in xrange(count):
			K[i][j] = kernel(xs[0,i],xs[0,j])

	var = np.var(xs)

	I = np.identity(count)

	k_dot_iden = np.linalg.inv(np.dot(var,I) + K)

	intermediate_mu = np.dot(k_x, k_dot_iden)

	mu = np.dot(intermediate_mu, np.transpose(ys))

	term_one  = np.dot(k_x, k_dot_iden)

	inside_bracket = 1 - np.dot(term_one, np.transpose(k_x))

	std = 2 * (inside_bracket ** .5) 

	return mu, std

def question_b(xs):

	mu_list = []
	std_neg = []
	std_pos = []

	for x in xs[0]:

		mu, std = mu_std(x)

		mu_list.append(mu)

		std_pos.append(mu + std)

		std_neg.append(mu - std)

	return mu_list, std_pos, std_neg


def flatten_me(list):
	return np.asarray(list).flatten()

def gen_picture_b():

	m_list, s_list, n_list =  question_b(xs)
	plt.scatter(xs[0], flatten_me(m_list), color='r' )
	plt.scatter(xs[0], flatten_me(s_list), color = 'g')
	plt.scatter(xs[0], flatten_me(n_list), color='b' )

	plt.show()


def posterior_c():

	zs  = interval(3)


	f_z_list = []
	f_z_list.append(normal(0,1))

	for cnt in xrange(1,34):
		z_cnt = zs[cnt]

		K = np.ones((100+cnt, 100+cnt))
		for i in xrange(100):
			for j in xrange(100):
				K[i][j] = kernel(xs[0,i], xs[0,j])
		for i in xrange(cnt):
			for j in xrange(cnt):
				K[100+i][100+j] = kernel(zs[i], zs[j])
		
		K = K + np.dot(4 , np.identity(100+cnt))
		left = np.asarray([xs[0]])
		right = np.asarray([zs[:cnt]])

		concated = np.concatenate((left,right), axis = 1)

		k_x = np.asarray([[kernel(z_cnt, x) for x in concated[0]]])

		var_local = np.var(xs)

		I = np.identity(100+cnt)

		right = list(ys[0])
		ys_local = np.asarray([right+f_z_list])


		k_dot_iden = np.linalg.inv(np.dot(var_local,I) + K)

		intermediate_mu = np.dot(k_x, k_dot_iden)
		
 		mu = np.dot( intermediate_mu, np.transpose(ys_local))
		var_first_term = np.dot(intermediate_mu, np.transpose(k_x))
		

		var = 1 - var_first_term
		

		f_z_list.append(normal(mu, var ** .5))
	return zs, f_z_list


def show_graph_c():

	COLOR = itertools.cycle(colors.cnames)
	for _ in xrange(20):
		x,y = posterior_c()
		plt.scatter(x,y,color=next(COLOR))

	plt.show()

show_graph_c()



