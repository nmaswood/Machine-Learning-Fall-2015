import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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


def knn(k, training_data, labels, test_data):

	knn_classifications  = []
	for test_point in test_data:
		distance_list = []
		for index, training_point in enumerate(training_data):
			distance = np.linalg.norm(test_point - training_point)
			distance_list.append((labels[index], distance))
		
		#sort distance tuples according to distance ascending order
		sorted_distance_list = sorted(distance_list, key=lambda distance_tuple: distance_tuple[1]) 
		
		#k closest distance tuples
		k_distance_list  = sorted_distance_list[:k]

		#find most common label amongst k closest
		counted_labels = Counter(label[0] for label in k_distance_list)
		most_common = counted_labels.most_common()

		
		for lable_freq_tuple in most_common:
			if(lable_freq_tuple[1] != most_common[0][1]):
				most_common.remove(lable_freq_tuple)
		
		most_common_label_list = []
		
		for label_freq_tuple in most_common:
			most_common_label_list.append(label_freq_tuple[0])

		#classify	
		if len(most_common_label_list) == 1:
			knn_classifications.append(most_common_label_list[0])

		#handle ties (if there is a tie, pick the closest), classify	
		else:
			for label_dist_tuple in k_distance_list:
				if label_dist_tuple[0] in most_common_label_list:
					knn_classifications.append(label_dist_tuple[0])
					break
	return knn_classifications

#2-class case
#knn_1 = knn(4, train2k_databw_35, train2k_label_35, test200_databw_35)
#np.savetxt('knn.label.test200.txt', np.array(knn_1) )

#multi-class case
#knn_2 = knn(4, train5k_databw_01234, train5k_label_01234, test500_databw_01234)
#np.savetxt('knn.label.test500.txt', np.array(knn_2) )



















