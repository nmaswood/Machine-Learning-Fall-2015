import numpy as np 
from numpy import cumsum, matlib
from  glob import glob
from scipy import misc
from multiprocessing import Pool
from functools import partial
from math import log, exp
import hickle as hkl

steps = 8
Nfeatures = 295937/ steps


def getImages(numOfPictures,row=64, col=64, save=False):

	pos_int_image_list = []
	neg_int_image_list = []


	files = glob("./"+ "faces"+ str(numOfPictures) + "/*.jpg")

	for i in xrange (numOfPictures):
			imgGray = misc.imread (files[i], flatten=1) 
			intImg = np.zeros((row+1,col+1))
			intImg [1:row+1, 1:col+1] = np.cumsum(cumsum(imgGray,axis=0),axis=1)
			pos_int_image_list.append(intImg)

	files = glob("./"+ "background"+ str(numOfPictures)+ "/*.jpg")

	for i in xrange (numOfPictures):
			imgGray = misc.imread (files[i], flatten=1)
			intImg = np.zeros((row+1,col+1))
			intImg [1:row+1,1:col+1] = np.cumsum(cumsum(imgGray,axis=0),axis=1)
			neg_int_image_list.append(intImg)
	total_int_image_list = pos_int_image_list + neg_int_image_list
	if save:
		hkl.dump(total_int_image_list, 'images'+str(numOfPictures) + ".hkl", mode='w')

	return (pos_int_image_list, neg_int_image_list)

def getHaar (numOfPictures, row=64, col=64, npSave = False):
	p = Pool()

	Nimg = numOfPictures * 2

	print "getting images"

	features = np.zeros((Nfeatures, Nimg))

	partial_func = partial(computeFeature, row, col)


	total_int_image_list = getImages(numOfPictures,row,col)
	pos_int_image_list = total_int_image_list[0]
	neg_int_image_list =  total_int_image_list[1]

	print "mapping 1st half"

	VALUES = p.map(partial_func, pos_int_image_list)

	for i in xrange(numOfPictures):
			features[:,i] = VALUES[i]
	
	print "mapping 2nd half"


	VALUES = p.map(partial_func, neg_int_image_list)

	for i in xrange(numOfPictures):
			features[:,i+ numOfPictures] = VALUES[i]

	if(npSave == True):
		hkl.dump(features, 'features'+str(numOfPictures) + ".hkl", mode='w')

	print "feat ", features[1000,:]
				
'''
Given four corner points in the integral image 
calculate the sum of pixels inside the rectangular. 
'''
def sumRect(I, rect_four): 
	
	row_start = rect_four[0]
	col_start = rect_four[1] 
	width = rect_four[2]
	height = rect_four[3] 
	one = I[row_start-1, col_start-1]
	two = I[row_start-1, col_start+width-1]
	three = I[row_start+height-1, col_start-1]
	four = I[row_start+height-1, col_start+width-1]
	return four + one -(two + three)

'''
Computes the features. The cnt variable can be used to count the features. 
If you'd like to have less or more features for debugging purposes, set the 
Nfeatures =cnt in getHaar(). 
'''
def computeFeature(row, col,I): 

	feature = np.zeros(Nfeatures)
	cnt = 0 
	
	window_h = steps; window_w= steps * 2 #window/feature size 
	for h in xrange(1,row/window_h+1): #extend the size of the rectangular feature
		for w in xrange(1,col/window_w+1):
			for i in xrange (1,row+1-h*window_h+1,4): #stride size=4
				for j in xrange(1,col+1-w*window_w+1,4): 
					rect1=np.array([i,j,w,h]) #4x1
					rect2=np.array([i,j+w,w,h])
					feature [cnt]=sumRect(I, rect2)- sumRect(I, rect1) 
					cnt += 1

	window_h = steps * 2; window_w= steps

	for h in xrange(1,row/window_h+1): 
		for w in xrange(1,col/window_w+1):
			for i in xrange (1,row+1-h*window_h+1,4):
				for j in xrange(1,col+1-w*window_w+1,4):
					rect1=np.array([i,j,w,h])
					rect2=np.array([i+h,j,w,h])
					feature[cnt]=sumRect(I, rect1)- sumRect(I, rect2)
					cnt+=1

	return feature 
	
def getWeakClassifier(features, weight, label, Npos):
	Nfeatures, Nimgs = features.shape
	currentMin = np.inf
	tPos = np.matlib.repmat(np.sum(weight[:Npos,0]), Nimgs,1) 
	tNeg = np.matlib.repmat(np.sum(weight[Npos:Nimgs,0]), Nimgs,1)
	
	for i in xrange(Nfeatures):
		#get one feature for all images
		oneFeature = features[i,:]

		# sort feature to thresh for postive and negative
		sortedFeature = np.sort(oneFeature)
		sortedIdx = np.argsort(oneFeature)
	
		# sort weights and labels
		sortedWeight = weight[sortedIdx]
		sortedLabel = label[sortedIdx]
		
		# compute the weighted errors 
		sPos = cumsum(np.multiply(sortedWeight,sortedLabel)) 
		sNeg = cumsum(sortedWeight)- sPos
		
		sPos = sPos.reshape(sPos.shape[0],1)
		sNeg = sNeg.reshape(sNeg.shape[0],1)
		errPos = sPos + (tNeg -sNeg)
		errNeg = sNeg + (tPos -sPos)
	
		# choose the threshold with the smallest error
		allErrMin = np.minimum(errPos, errNeg) # pointwise min
		
		errMin = np.min(allErrMin)
		idxMin = np.argmin(allErrMin)
		
		# classification result under best threshold
		result = np.zeros((Nimgs,1))
		if (errPos [idxMin] <= errNeg[idxMin]):
			p = -1
			end = result.shape[0]
			result[idxMin:end] = 1
			result[sortedIdx] = np.copy(result)
		else:
			p = 1
			result[:idxMin] = 1
			result[sortedIdx] = np.copy(result)

		#get the parameters that minimize the classification error
		if (errMin < currentMin):
			currentMin = errMin
			if (idxMin==1):
				theta = sortedFeature[1] - 0.5
			elif (idxMin==Nfeatures):
				theta = sortedFeature[Nfeatures] + 0.5
			else:
				theta = (sortedFeature[idxMin]+sortedFeature[idxMin - 1])/2
			polarity = p
			featureIdx = i
			bestResult = result
	return currentMin, theta, polarity, featureIdx, bestResult

#getHaar(200, 64, 64, npSave=True)

