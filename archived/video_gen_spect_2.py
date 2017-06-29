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

us_frames = glob('Data/Unlabeled/lesion_c4/*.jpg')

us_frames = sorted_nicely(us_frames)

prefix = 'Data/Unlabeled/lesion_c4_raw_seg_combine/'

for i in range(len(us_frames)):

	try:

		img = Image.open(us_frames[i]).convert('L')
		img_arr = np.asarray(img)


		# Resize it to 10% of the original size to speed up the processing
		us = sp.misc.imresize(img_arr, 0.10) / 255.

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
		N_REGIONS = 8
		reg_range = np.arange(N_REGIONS)


		#for assign_labels in ('discretize'):
		#'kmeans'

		assign_labels = 'discretize'

		t0 = time.time()
		labels = spectral_clustering(graph, n_clusters=N_REGIONS,
									 assign_labels=assign_labels, random_state=1)


		#for j in range(10):
		#	print(labels[j*56:(j+1)*56])

		new_label_ordering = []

		for n in labels:

			if n not in new_label_ordering:

				new_label_ordering.append(n)

		for j in range(len(labels)):

			current_label = labels[j]
			new_label = reg_range[new_label_ordering.index(current_label)]

			labels[j] = new_label


		#print(labels.shape)

		t1 = time.time()
		labels = labels.reshape(us.shape)

		#print(labels.shape)
		#print(labels[:4,:4])

		# FIX COLOR MAP TO MAKE PRETTY VIDS
		'''
		for r in range(labels.shape[0]):

			regions = []

			for c in range(labels.shape[1]):

				if labels[r][c] not in regions:

					regions.append(labels[r][c])

			print(regions)
		'''

		plt.figure()#figsize=(5, 5))
		plt.subplot(1,2,1)

		plt.imshow(us, cmap=plt.cm.gray)
		plt.axis('off')

		for l in range(N_REGIONS):
			plt.contour(labels == l, contours=1,
						colors=[plt.cm.spectral(l / float(N_REGIONS))])

		plt.subplot(1,2,2)
		plt.imshow(img_arr, cmap=plt.cm.gray)

		plt.axis('off')

		plt.savefig(prefix + 'segments_' + str(i) + '.png', bbox_inches='tight')


		#plt.xticks(())
		#plt.yticks(())

		title = 'Spectral clustering: %s, %.2fs' % (assign_labels, (t1 - t0))

		print(title)
		plt.title(title)

		#x = input(">")

		#plt.show()
	except:
		print('error')
		pass
