import numpy as np 
from numpy import cumsum, matlib
from  glob import glob
from scipy import misc
from multiprocessing import Pool
from functools import partial
from math import log, exp, sqrt, copysign, floor
import hickle as hkl
from violoajones import getImages, getHaar, sumRect, computeFeature, getWeakClassifier
from collections import OrderedDict
import cv2

steps = 8
Nfeatures = 295937 / steps
pixel_spacing = 20
wcnum = 250

#By Sohini Upadhyay & Nasr Maswood

def getIntegralImages (row, col):

	integral_image_list = []
	coordinate_list = []

	imgGray = misc.imread ("class.jpg", flatten=1) #array of floats, gray scale image
	intImg = np.zeros((row+1,col+1))
	intImg [1:row+1,1:col+1] = np.cumsum(cumsum(imgGray,axis=0),axis=1)	

	cnt = 0 
	for j in xrange(1,row - 64,pixel_spacing):
		for k in xrange(1,col -64,pixel_spacing):
			sliced_array = intImg[j:j + 65,k:k + 65]
			integral_image_list.append(sliced_array)
			coordinate_list.append((k,j))
			cnt += 1

	return integral_image_list, coordinate_list

def reweigh(Npos):

	total_images = Npos * 2

	features = hkl.load('features'+str(Npos)+'.hkl')

	label = np.zeros((total_images, 1))

	label[:,0] = [1]*Npos + [0]*Npos

	weight = np.ones((total_images,1)) / total_images

	feature_index_list = []
	alpha_list = []
	theta_list = []
	polarity_list = []
	best_result_list = []

	for t in xrange(wcnum):

		currentMin, theta, polarity, featureIdx, bestResult = getWeakClassifier(features,weight,label,Npos)

		alpha = log((1 - currentMin)/currentMin) / 2.0
		Z = 2.0 * sqrt( currentMin * ( 1.0 - currentMin))


		feature_index_list.append(featureIdx)
		alpha_list.append(alpha)
		theta_list.append(theta)
		polarity_list.append(polarity)
		best_result_list.append(bestResult)
		
		print "---"
		print "t", t
		print "featureIdx", featureIdx


		for i in xrange(total_images):

			weight[i,0] =  (weight[i,0] * exp(-1 *  alpha * label[i] * bestResult[i]))/ Z


	hkl.dump(feature_index_list,'feature_index_list' + str(Npos) + ".hkl")
	hkl.dump(alpha_list,'alpha_list' + str(Npos) + ".hkl")
	hkl.dump(theta_list,'theta_list' + str(Npos) + ".hkl")
	hkl.dump(polarity_list,'polarity_list' + str(Npos) + ".hkl")
	hkl.dump(best_result_list,'best_result_list' + str(Npos) + ".hkl")


def getWeakClassifiers(Npos):

	feature_index_list = hkl.load('feature_index_list' + str(Npos) + ".hkl")
	alpha_list = hkl.load('alpha_list' + str(Npos) + ".hkl")
	theta_list = hkl.load('theta_list' + str(Npos) + ".hkl")
	polarity_list = hkl.load('polarity_list' + str(Npos) + ".hkl")
	best_result_list = hkl.load('best_result_list' + str(Npos) + ".hkl")

	return feature_index_list, alpha_list, theta_list, polarity_list, best_result_list

feature_index_list_200, alpha_list_200, theta_list_200, polarity_list_200, best_result_list_200 =  getWeakClassifiers(200)

def findTheta(Npos,alpha_list, best_result_list,imageIndex):

	labels = np.zeros((Npos*2, 1))

	labels[:,0] = [1]*Npos + [0]*Npos

	theta = 0
	num_list = [alpha_list[i] * best_result_list[i][imageIndex] for i in xrange(wcnum)]

	theta_max = sum(num_list)

	for _ in xrange(int(theta_max)):

		sign = copysign(1, theta_max - theta)

		if labels[imageIndex] == sign:
			theta -=1
			return theta
		else:
			theta+=1
	return theta


def classFeatures():
	features = hkl.load("class-features.hkl")
	return features

def findTheta_iterative(Npos,alpha_list, best_result_list):
	theta_min = findTheta(Npos,alpha_list, best_result_list,0)
	Nimg = Npos * 2
	for i in xrange(1, Nimg):
		theta_min_curr = findTheta(Npos, alpha_list, best_result_list, i)
		if theta_min_curr < theta_min:
			theta_min = theta_min_curr
	return theta_min

def index_to_haar(featureIdx, I):

	half_features = Nfeatures/2
	cnt =0 

	if featureIdx < half_features:
		window_h = steps; window_w = 2 * steps
	else: 
		window_h = 2 * steps; window_w = steps

	row = 64; col = 64
	for h in xrange(1,row/window_h+1): #extend the size of the rectangular feature
		for w in xrange(1,col/window_w+1):
			for i in xrange (1,row+1-h*window_h+1,2): #stride size=4
				for j in xrange(1,col+1-w*window_w+1,2): 
					if(cnt == featureIdx):
						rect1=np.array([i,j,w,h]) #4x1
						rect2=np.array([i,j+w,w,h])
						return sumRect(I, rect2)- sumRect(I, rect1)
	
					cnt += 1
	
def buildStrong(I, Npos, wcnum_choose):
	
	feature_index_list, alpha_list, theta_list, polarity_list, best_result_list = getWeakClassifiers(Npos)

	big_theta = findTheta_iterative(Npos, alpha_list, best_result_list)

	wc_list = []
	for index, value in enumerate(feature_index_list[:wcnum_choose]):
		polarity = -polarity_list[index]
		haar = index_to_haar(value, I)
		theta = theta_list[index]
		wc_list.append(copysign(1, polarity*(haar - theta)))

	#num_list = [alpha_list[i] * wc_list[i] for idx, val in enumerate(feature_index_list[:wcnum_choose])]
	num_list = []
	for i in xrange(wcnum_choose):
		alpha = alpha_list[i]
		wc = wc_list[i]
		num_list.append(alpha*wc)

	the_sum = sum(num_list)
	return copysign(1, the_sum - big_theta)

def cascade(I, Npos, numStages):
	delta_wcnum = 3
	for _ in xrange(numStages):
		strong_result = buildStrong(I, Npos, delta_wcnum)

		if strong_result == -1:
			return strong_result
		delta_wcnum += 3
	return strong_result

class_numStages = 5

def scanImage(img,numStages,Npos):
	return_coordinate_list = []
	integral_image_list, coordinate_list = getIntegralImages(1280, 1600)
	for idx, I in enumerate(integral_image_list):
		is_face = cascade(I, Npos, numStages)
		k,j = coordinate_list[idx]
		if is_face == 1:
			#cv2.rectangle(img, (k,j), (k+64, j+64), (255,0,0), 3)
			return_coordinate_list.append((k,j))
		print idx

	hkl.dump(return_coordinate_list,'return_coordinate_list'+ str(class_numStages) +'.hkl')

	return return_coordinate_list
		

class_Npos = 200
img =misc.imread ("class.jpg", flatten=1)
img = img.astype('uint8')
scanImage(img,class_numStages, class_Npos)
#coordinate_list = hkl.load("return_coordinate_list.hkl")

"""
def no_overlaps(coordinate_list):
	new_coord_list = []
	for i in range(len(coordinate_list)):
		if(i + 30 > len(coordinate_list)):
			break
		print "theLen", len(coordinate_list)
		print "i", i
		if(abs(coordinate_list[i][0] - coordinate_list[i+1][0]) < .1):
			del coordinate_list[i]
		if(abs(coordinate_list[i][1] - coordinate_list[i+1][1]) < .1):
			del coordinate_list[i]

	return coordinate_list



		for k,j in new_coord_list:
		cv2.rectangle(img, (k,j), (k+64, j-64), (255,0,0), 3)
	cv2.imshow("img",img)
	cv2.waitKey()
	print "BAD",new_coord_list
	return new_coord_list

for _ in xrange(1):
	coordinate_list = no_overlaps(coordinate_list)

print len(coordinate_list)


for k,j in coordinate_list:
	print k,j
	cv2.rectangle(img, (k,j), (k+64, j-64), (255,0,0), 3)
cv2.imshow("img",img)
cv2.waitKey()


"""


