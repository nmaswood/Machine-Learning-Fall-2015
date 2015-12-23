import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm


#2-class files
train2k_databw_35= np.genfromtxt("train2k.databw.35")
train2k_label_35= np.genfromtxt("train2k.label.35")
test200_databw_35= np.genfromtxt("test200.databw.35")

#Multi-class files
train5k_databw_01234= np.genfromtxt("train5k.databw.01234")
train5k_label_01234= np.genfromtxt("train5k.label.01234")
test500_databw_01234= np.genfromtxt("test500.databw.01234")

#Debugging display tool
#plt.imshow(test200_databw_35[7].reshape(28,28), cmap=cm.Greys_r)
#plt.show()

def minus1_to_zero(labels):
	for i in range(len(labels)):
		if labels[i] == -1:
			labels[i] = 0
	return labels

def zero_to_minus1(labels):
	for i in range(len(labels)):
		if labels[i] == 0:
			labels[i] = -1
	return labels

def perceptron(M, training_data, labels, testing_data):

	indiv_labels = Counter(labels)
	k = len(indiv_labels)
	weight_matrix = np.zeros( (k, len(training_data[0])) )

	perceptron_classifications = []

	#runs algorithm M times
	for _ in range(M):
		mistakes = 0

		#uses training data to construct correct weights
		for training_index, training_point in enumerate(training_data):
			training_dot_product_list = []

			for weight_index in xrange(k):
				training_dot_product = np.dot(training_point, weight_matrix[weight_index])
				training_dot_product_list.append((weight_index,training_dot_product))

			training_sorted_dot_product_list = sorted(training_dot_product_list, key=lambda dot_tuple: dot_tuple[1], reverse=True)
			training_predicted_label = training_sorted_dot_product_list[0][0]
			
			#updates weights upon mistake
			if training_predicted_label != labels[training_index]:
				mistakes += 1
				weight_matrix[training_predicted_label] -= training_point/2
				weight_matrix[labels[training_index]] += training_point/2
	
	#classifies test data
	for test_index, test_point in enumerate(testing_data):
		test_dot_product_list = []

		for weight_index in xrange(k):
			test_dot_product = np.dot(test_point,weight_matrix[weight_index])
			test_dot_product_list.append((weight_index,test_dot_product))

		test_sorted_dot_product_list = sorted(test_dot_product_list, key=lambda dot_tuple: dot_tuple[1], reverse=True)
		test_predicted_label = test_sorted_dot_product_list[0][0]
		
		perceptron_classifications.append(test_predicted_label)

	return (perceptron_classifications, mistakes)



#2-class case
perceptron_1 = zero_to_minus1(perceptron(100, train2k_databw_35, minus1_to_zero(train2k_label_35), test200_databw_35)[0])
np.savetxt('perceptron.label.test200.txt', np.array(perceptron_1) )

#multi-class case
perceptron_2 = perceptron(100, train5k_databw_01234, train5k_label_01234, test500_databw_01234)[0]
np.savetxt('perceptron.label.test500.txt', np.array(perceptron_2) )


