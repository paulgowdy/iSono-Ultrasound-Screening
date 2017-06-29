from PIL import Image
from glob import glob
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version

import re 

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

us_frames = glob('Data/Segmentation/*.jpg')

us_frames = sorted_nicely(us_frames)

masks = glob('Data/Segmentation/masks/*.jpg')

masks = sorted_nicely(masks)

#prefix = 'Data/Segmentation/spectral_segs/'

for i in range(len(us_frames)):



	img = Image.open(us_frames[i]).convert('L')
	img_arr = np.asarray(img)

	mask = Image.open(masks[i]).convert('L')
	mask = np.asarray(mask)

	#print(np.amax(mask), np.argmax(mask))
	#print(mask.shape)

	#print(mask[200,200])

	#plt.imshow(mask)
	#plt.show()

	# Resize it to 10% of the original size to speed up the processing
	us = sp.misc.imresize(img_arr, 0.10) / 255.

	img_arr = sp.misc.imresize(img_arr, 0.10) 
	mask = sp.misc.imresize(mask, 0.10)



	# Convert the image into a graph with the value of the gradient on the
	# edges.
	graph = image.img_to_graph(us)

	# Take a decreasing function of the gradient: an exponential
	# The smaller beta is, the more independent the segmentation is of the
	# actual image. For beta=1, the segmentation is close to a voronoi
	beta = 5
	eps = 1e-6
	graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

	# Apply spectral clustering (this step goes much faster if you have pyamg
	# installed)
	N_REGIONS = 25
	reg_range = np.arange(N_REGIONS)

	assign_labels = 'discretize'

	#t0 = time.time()

	labels = spectral_clustering(graph, n_clusters=N_REGIONS,
								 assign_labels=assign_labels, random_state=1)

	new_label_ordering = []

	for n in labels:

		if n not in new_label_ordering:

			new_label_ordering.append(n)

	for j in range(len(labels)):

		current_label = labels[j]
		new_label = reg_range[new_label_ordering.index(current_label)]

		labels[j] = new_label

	#t1 = time.time()
	labels = labels.reshape(us.shape)

	segment_stats = []

	for n in range(N_REGIONS):

		segment_stats.append([n,0,0,0]) 

	#print(labels.shape)

	'''
	labels = sp.misc.imresize(labels, (img_arr.shape))
	labels = np.clip(labels, 0, 24)



	print(labels.shape)
	print(img_arr.shape)
	print(mask.shape)

	plt.figure()#figsize=(5, 5))
	plt.subplot(1,3,1)

	plt.imshow(us, cmap=plt.cm.gray)
	plt.axis('off')

	for l in range(N_REGIONS):
		plt.contour(labels == l, contours=1,
					colors=[plt.cm.spectral(l / float(N_REGIONS))])

	plt.subplot(1,3,2)
	plt.imshow(img_arr, cmap=plt.cm.gray)

	mask = Image.open(masks[i]).convert('L')
	mask = np.asarray(mask)

	plt.subplot(1,3,3)
	plt.imshow(mask, cmap=plt.cm.gray)

	plt.axis('off')
	plt.show()

	'''
	for r in range(img_arr.shape[0]):

		for c in range(img_arr.shape[1]):

			#print(r,c, mask[r,c])

			current_label = labels[r][c]

			#print(current_label)

			segment_stats[current_label][1] += 1
			segment_stats[current_label][2] += np.ceil(mask[r,c]/float(255))
			segment_stats[current_label][3] += img_arr[r][c]

	mask_and_labels = np.zeros_like(labels)

	print(mask_and_labels.shape)
	print(np.amax(mask_and_labels))

	for r in range(img_arr.shape[0]):

		for c in range(img_arr.shape[1]):

			current_label = labels[r][c]

			mask_and_labels[r,c] += 255*segment_stats[current_label][2]/float(segment_stats[current_label][1])

	print(mask_and_labels.shape)
	print(np.amax(mask_and_labels))


	for s in segment_stats:
		print(s)

	plt.figure()#figsize=(5, 5))
	plt.subplot(1,3,1)

	#plt.imshow(us, cmap=plt.cm.gray)
	plt.imshow(mask_and_labels)
	plt.axis('off')


	#for l in range(N_REGIONS):
	#	plt.contour(labels == l, contours=1,
	#				colors=[plt.cm.spectral(l / float(N_REGIONS))])

	plt.subplot(1,3,2)
	plt.imshow(img_arr, cmap=plt.cm.gray)

	mask = Image.open(masks[i]).convert('L')
	mask = np.asarray(mask)

	plt.subplot(1,3,3)
	plt.imshow(mask, cmap=plt.cm.gray)

	plt.axis('off')
	plt.show()

	print('-------')





