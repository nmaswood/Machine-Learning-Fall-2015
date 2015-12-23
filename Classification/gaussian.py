import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import log

#2-class files
train2k_databw_35= np.genfromtxt("train2k.databw.35")
test200_databw_35= np.genfromtxt("test200.databw.35")
train2k_label_35= np.genfromtxt("train2k.label.35")

#Multi-class files
train5k_databw_01234= np.genfromtxt("train5k.databw.01234")
train5k_label_01234= np.genfromtxt("train5k.label.01234")
test500_databw_01234= np.genfromtxt("test500.databw.01234")

#Debugging display tool
#plt.imshow(test200_databw_35[7].reshape(28,28), cmap=cm.Greys_r)
#plt.show()

def R(x, mean, cov,label_probability):
	cov = cov + .01 * np.identity(cov.shape[0])
	log_pi = log(label_probability)
	det =  np.linalg.eig(cov)[0].sum()
	term2 = log(abs(det)) / 2.0
	term3_1 = (x - mean).T
	term3_2= np.dot(np.linalg.inv(cov),((x - mean) / 2.0))
	return log_pi - term2 - np.dot(term3_1, term3_2)

def gaussian(training_data, labels, testing_data):
	indiv_labels = Counter(labels)
	total_labels = float(len(labels))
	
	gaussian_classification = []

	#key=label, value= image vectors with that label
	label_vector_dictionary  = {}

	#init dictionary
	for label in labels:
		label_vector_dictionary[label] = []
	
	#fill dictionary
	for i in range(len(labels)):
		label_vector_dictionary[labels[i]].append(training_data[i])

	#key = label, value = mean, cov corresponding to the vectors with that label
	label_gauss_dict = {}

	for key, value in label_vector_dictionary.items():
		cov = np.cov(np.array(value).T)
		mean = np.mean(value, axis=0)

		label_gauss_dict[key] = (cov, mean)

	for test_point in testing_data:
		#list of tuples = (label, probability point has that label)
		label_probability_list = []

		for label in label_gauss_dict:
			cov = label_gauss_dict[label][0]
			mean = label_gauss_dict[label][1]

			label_probability = indiv_labels[label] / total_labels
			label_probability_list.append((label, R(test_point, mean, cov, label_probability)))

		sorted_label_probability_list = sorted(label_probability_list, key=lambda probability_tuple: probability_tuple[1], reverse=True)
		correct_label = sorted_label_probability_list[0][0] 

		gaussian_classification.append(correct_label)

	return gaussian_classification
	



#2-class case
#gauss_1 = gaussian(train2k_databw_35, train2k_label_35, test200_databw_35)
#np.savetxt('mixture.label.test200.txt', np.array(gauss_1) )

#multi-class case
#gauss_2 = gaussian(train5k_databw_01234, train5k_label_01234, test500_databw_01234)
#np.savetxt('mixture.label.test500.txt', np.array(gauss_2) )




