from PIL import Image
from glob import glob
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy.ma as ma
from statsmodels import robust
import csv
import re 
import pickle

from random import shuffle

import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

import xgboost
from xgboost import XGBClassifier, XGBRegressor

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def is_float(val):
        try:
            float(val)
        except ValueError:
            return False
        else:
            return True

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def nan_cropper(a):

	nans = np.isnan(a)
	nancols = np.all(nans, axis=0) # 10 booleans, True where col is all NAN
	nanrows = np.all(nans, axis=1) # 15 booleans

	firstcol = nancols.argmin() # 5, the first index where not NAN
	firstrow = nanrows.argmin() # 7

	lastcol = len(nancols) - nancols[::-1].argmin() # 8, last index where not NAN
	lastrow = len(nanrows) - nanrows[::-1].argmin() # 10

	return a[firstrow:lastrow,firstcol:lastcol]

'''
# Big function - does all the work	
def segment_stats_collector_test(img_arr, labels, n_segments):

	

	CANT HAVE ANYTHING WITH the MASK!

	segment stats: a list of dictionaries
	one dictionary for each segment containing the following descriptive stats (in this order)

	.segment #
	.centroid x-coord
	.centroid y-coord
	.total area (pixels)
	.max x dim of bounding box
	.max y dim of bounding box
	.(max x dim)/total area
	.(max y dim)/total area
	.(max x dim)/(max y dim)
	.total pixel intensity
	.mean pixel intensity
	.min pixel intensity
	.max pixel intensity
	.pixel intensity range

	.standard deviation of pixels
	.mean absolute deviation of pixels
	.scipy.stats.skew
	.scipy.stats.kurtosis

	.# of edges

	

	.[one for each other segment]
	.for each other segment, 1 if you border it, 0 if you dont
	.the difference in mean pixel intensity between current segment and each other segment
	.product of the two above

	.mean difference with bordering segments

	---

	Not implemented yet

	histogram of gradients from within region
	gradients along edge (identify edge pixels)
	longest actual x and y dim, not bounding box
	how central the region is in the image as a %, so like centroid x,y div the image length/width...


	
	#if img_arr.shape != mask.shape:

	#	mask = sp.misc.imresize(mask, img_arr.shape)

	segment_stats = []

	for n in range(n_segments):

		segment_stats.append({
			'segment_id': n,
			'centroid_x': 0.0,
			'centroid_y': 0.0,
			'total_area': 0,
			'bb_x_dim': 0,
			'bb_y_dim': 0,
			'bb_x_div_total_area': 0.0,
			'bb_y_div_total_area': 0.0,
			'bb_x_div_bb_y': 0.0,
			'total_pixel_intensity': 0,
			'mean_pixel_intensity': 0.0,
			'min_pix': 0,
			'max_pix': 0,
			'pix_range': 0,
			'std': 0.0,
			'mad': 0.0,
			'skew': 0.0,
			'kurtosis': 0.0,
			'bordering_segs_list': [],
			'n_borders': 0,
			'one_hot_borders': [],
			'seg_mean_differences': [],
			'seg_mean_diff_if_bordering': [],
			'mean_border_diff': 0.0,
			'mean_border_abs_diff': 0.0
			}) 


	centroids = ndimage.measurements.center_of_mass(labels,labels,range(n_segments))

	# Loop over every pixel
	for r in range(img_arr.shape[0]):

		for c in range(img_arr.shape[1]):

			current_label = labels[r][c]

			bordering = []

			for ri in [-1,0,1]:

				for ci in [-1,0,1]:

					try:

						potential_border = labels[r + ri][c + ci]

						if potential_border != current_label and potential_border not in bordering:

							bordering.append(potential_border)

					except:

						pass
			

			# total pixels in the region
			segment_stats[current_label]['total_area'] += 1

			

			# total pixel value in the region
			segment_stats[current_label]['total_pixel_intensity'] += 255 * img_arr[r][c]

			# List of all neighboring segments
			for b in bordering:

				if b not in segment_stats[current_label]['bordering_segs_list']:

					segment_stats[current_label]['bordering_segs_list'].append(b)


	# Loop over segments
	for s in segment_stats:

		p = segment_stats.index(s)

		# Number of borders
		s['n_borders'] = len(s['bordering_segs_list'])

		# Isolate segment
		segment_mask = ma.masked_not_equal(labels,p)
		m2 = np.multiply(img_arr, segment_mask/float(p))
		cr_m2 = 255 * nan_cropper(m2)

		# remove all masked pixels and flatten
		z = cr_m2.compressed()

		# Measures of variance
		s['std'] = np.std(cr_m2)
		s['mad'] = mad(cr_m2)
		s['skew'] = sp.stats.skew(z)
		s['kurtosis'] = sp.stats.kurtosis(z)

		# Bounding box stats
		s['bb_x_dim'] = cr_m2.shape[1]
		s['bb_y_dim'] = cr_m2.shape[0]

		# Segment min and max
		s['min_pix'] = np.amin(cr_m2)
		s['max_pix'] = np.amax(cr_m2)
		s['pix_range'] = s['max_pix'] - s['min_pix']

		# BB Dimension secondary stats
		if s['total_area'] == 0:
			s['total_area'] = 1

		s['bb_x_div_total_area'] = s['bb_x_dim']/float(s['total_area'])
		s['bb_y_div_total_area'] = s['bb_y_dim']/float(s['total_area'])
		s['bb_x_div_bb_y'] = s['bb_x_dim']/float(s['bb_y_dim'])

		# Centroid x and y
		cent = centroids[p]
		s['centroid_x'] = cent[1]
		s['centroid_y'] = cent[0]

		# Mean pixel intensity
		s['mean_pixel_intensity'] = s['total_pixel_intensity']/float(s['total_area'])

	for s in segment_stats:

		for j in range(n_segments):

			if j in s['bordering_segs_list']:

				s['one_hot_borders'].append(1)

			else:

				s['one_hot_borders'].append(0)

			seg_mean_diff = s['mean_pixel_intensity'] - segment_stats[j]['mean_pixel_intensity']

			s['seg_mean_differences'].append(seg_mean_diff)

			s['seg_mean_diff_if_bordering'].append(s['seg_mean_differences'][-1] * s['one_hot_borders'][-1])

	for s in segment_stats:

		m = np.sum(s['seg_mean_diff_if_bordering'])

		s['mean_border_diff'] = m/float(s['n_borders'])

		n = np.sum(np.abs(s['seg_mean_diff_if_bordering']))

		s['mean_border_abs_diff'] = n/float(s['n_borders'])

	return segment_stats

'''

# Big function - does all the work	
def new_segment_stats_collector_test(img_arr, labels, n_segments):

	'''
	segment stats: a list of dictionaries
	one dictionary for each segment containing the following descriptive stats (in this order)

	.segment #
	.centroid x-coord
	.centroid y-coord
	.total area (pixels)
	.max x dim of bounding box
	.max y dim of bounding box
	.(max x dim)/total area
	.(max y dim)/total area
	.(max x dim)/(max y dim)
	.total pixel intensity
	.mean pixel intensity
	.min pixel intensity
	.max pixel intensity
	.pixel intensity range

	.standard deviation of pixels
	.mean absolute deviation of pixels
	.scipy.stats.skew
	.scipy.stats.kurtosis

	.# of edges

	.MASK PIXELS contained in the segment
	.(mask pixels)/(total pixels)

	.[one for each other segment]
	.for each other segment, 1 if you border it, 0 if you dont
	.the difference in mean pixel intensity between current segment and each other segment
	.product of the two above

	.mean difference with bordering segments

	EDGES
	.total edge-pix intensity along edges
	mean edge-pix intensity along edges
	.total edge-pix intensity in region (non edges)
	mean edge-pix intensity in region (non edges)
	.total edge-pix intensity in region (edges and non edges)
	mean edge-pix intensity in region (edges and non edges)

	---

	Not implemented yet

	histogram of gradients from within region
	gradients along edge (identify edge pixels)
	longest actual x and y dim, not bounding box
	how central the region is in the image as a %, so like centroid x,y div the image length/width...


	'''
	#if img_arr.shape != mask.shape:

	#	mask = sp.misc.imresize(mask, img_arr.shape)

	segment_stats = []

	for n in range(n_segments):

		segment_stats.append({
			'segment_id': n,
			'centroid_x': 0.0,
			'centroid_y': 0.0,
			'total_area': 0,
			'bb_x_dim': 0,
			'bb_y_dim': 0,
			'bb_x_div_total_area': 0.0,
			'bb_y_div_total_area': 0.0,
			'bb_x_div_bb_y': 0.0,
			'total_pixel_intensity': 0,
			'mean_pixel_intensity': 0.0,
			'min_pix': 0,
			'max_pix': 0,
			'pix_range': 0,
			'std': 0.0,
			'mad': 0.0,
			'skew': 0.0,
			'kurtosis': 0.0,
			'bordering_segs_list': [],
			'n_borders': 0,
			#'mask_count': 0,
			#'frac_masked': 0.0,
			'one_hot_borders': [],
			'seg_mean_differences': [],
			'seg_mean_diff_if_bordering': [],
			'mean_border_diff': 0.0,
			'mean_border_abs_diff': 0.0,

			'tot_edge_pix_intensity_edges': 0,
			'tot_edge_pix_intensity_nonedges': 0,
			'tot_edge_pix_intensity_both': 0,
			'mean_edge_pix_intensity_edges': 0.0,
			'mean_edge_pix_intensity_nonedges': 0.0,
			'mean_edge_pix_intensity_both': 0.0,

			'edge_count': 0
			}) 


	centroids = ndimage.measurements.center_of_mass(labels,labels,range(n_segments))

	seg_edge_img = np.zeros_like(img_arr)

	edge_img = edge_viewer(img_arr)

	# Loop over every pixel
	for r in range(img_arr.shape[0]):

		for c in range(img_arr.shape[1]):

			current_label = labels[r][c]

			bordering = []

			for ri in [-1,0,1]:

				for ci in [-1,0,1]:

					try:

						potential_border = labels[r + ri][c + ci]

						if potential_border != current_label and potential_border not in bordering:

							bordering.append(potential_border)



						if current_label != labels[r + ri][c + ci]:

							seg_edge_img[r,c] = 1

							segment_stats[current_label]['edge_count'] += 1


							#segment_stats[current_label]['tot_edge_pix_intensity_edges'] += edge_img[r,c]

					except:

						pass

			if seg_edge_img[r,c] == 1:

				segment_stats[current_label]['tot_edge_pix_intensity_edges'] += edge_img[r,c]

			else:

				segment_stats[current_label]['tot_edge_pix_intensity_nonedges'] += edge_img[r,c]


			

			# total pixels in the region
			segment_stats[current_label]['total_area'] += 1

			# total mask pixels in the region
			

			#if mask[r,c] > 0:

				#print(current_label)
				
			#	segment_stats[current_label]['mask_count'] += 1

			# total pixel value in the region
			segment_stats[current_label]['total_pixel_intensity'] += 255 * img_arr[r][c]

			# List of all neighboring segments
			for b in bordering:

				if b not in segment_stats[current_label]['bordering_segs_list']:

					segment_stats[current_label]['bordering_segs_list'].append(b)

	

	'''
	plt.figure()
	plt.imshow(edge_img, cmap=plt.cm.gray)
	plt.axis('off')
	plt.title('sobel edges')

	plt.figure()
	plt.imshow(seg_edge_img, cmap=plt.cm.gray)
	plt.axis('off')
	plt.title('segment edges')

	plt.show()
	'''

	# Loop over segments
	for s in segment_stats:

		p = segment_stats.index(s)

		s['tot_edge_pix_intensity_both'] = s['tot_edge_pix_intensity_nonedges'] + s['tot_edge_pix_intensity_edges']

		try:
			s['mean_edge_pix_intensity_edges'] = s['tot_edge_pix_intensity_edges']/float(s['edge_count'])
		except:
			s['mean_edge_pix_intensity_edges'] = 0

		try:
			s['mean_edge_pix_intensity_both'] = s['tot_edge_pix_intensity_both']/float(s['total_area'])
		except:
			s['mean_edge_pix_intensity_both'] = 0

		try:
			s['mean_edge_pix_intensity_nonedges'] = s['tot_edge_pix_intensity_nonedges']/float(s['total_area'] - s['edge_count'])
		except:
			s['mean_edge_pix_intensity_nonedges'] = 0

		# Number of borders
		s['n_borders'] = len(s['bordering_segs_list'])

		# Isolate segment
		segment_mask = ma.masked_not_equal(labels,p)
		m2 = np.multiply(img_arr, segment_mask/float(p))
		cr_m2 = 255 * nan_cropper(m2)

		# remove all masked pixels and flatten
		z = cr_m2.compressed()

		# Measures of variance
		s['std'] = np.std(cr_m2)
		s['mad'] = mad(cr_m2)
		s['skew'] = sp.stats.skew(z)
		s['kurtosis'] = sp.stats.kurtosis(z)

		# Bounding box stats
		s['bb_x_dim'] = cr_m2.shape[1]
		s['bb_y_dim'] = cr_m2.shape[0]

		# Segment min and max
		s['min_pix'] = np.amin(cr_m2)
		s['max_pix'] = np.amax(cr_m2)
		s['pix_range'] = s['max_pix'] - s['min_pix']

		# BB Dimension secondary stats
		if s['total_area'] == 0:
			s['total_area'] = 1

		s['bb_x_div_total_area'] = s['bb_x_dim']/float(s['total_area'])
		s['bb_y_div_total_area'] = s['bb_y_dim']/float(s['total_area'])
		s['bb_x_div_bb_y'] = s['bb_x_dim']/float(s['bb_y_dim'])

		# Centroid x and y
		cent = centroids[p]
		s['centroid_x'] = cent[1]
		s['centroid_y'] = cent[0]

		# Mean pixel intensity
		s['mean_pixel_intensity'] = s['total_pixel_intensity']/float(s['total_area'])

		# Masked pixel fraction
		#s['frac_masked'] = s['mask_count']/float(s['total_area'])

	for s in segment_stats:

		for j in range(n_segments):

			if j in s['bordering_segs_list']:

				s['one_hot_borders'].append(1)

			else:

				s['one_hot_borders'].append(0)

			seg_mean_diff = s['mean_pixel_intensity'] - segment_stats[j]['mean_pixel_intensity']

			s['seg_mean_differences'].append(seg_mean_diff)

			s['seg_mean_diff_if_bordering'].append(s['seg_mean_differences'][-1] * s['one_hot_borders'][-1])

	for s in segment_stats:

		m = np.sum(s['seg_mean_diff_if_bordering'])

		s['mean_border_diff'] = m/float(s['n_borders'])

		n = np.sum(np.abs(s['seg_mean_diff_if_bordering']))

		s['mean_border_abs_diff'] = n/float(s['n_borders'])

	return segment_stats

def spectral_segmentation(image_a, bet, N_REGIONS = 20):

	# Convert the image into a graph with the value of the gradient on the edges
	graph = image.img_to_graph(image_a)

	# Take a decreasing function of the gradient: an exponential
	# The smaller beta is, the more independent the segmentation is of the
	# actual image. For beta=1, the segmentation is close to a voronoi
	beta = bet
	eps = 1e-6
	graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

	# Apply spectral clustering
	reg_range = np.arange(N_REGIONS)

	assign_labels = 'discretize'

	#labels = spectral_clustering(graph, n_clusters=N_REGIONS,
	#							 assign_labels=assign_labels, random_state=1)

	labels = spectral_clustering(graph, n_clusters=N_REGIONS,
								  random_state=1, assign_labels=assign_labels)

	new_label_ordering = []

	for n in labels:

		if n not in new_label_ordering:

			new_label_ordering.append(n)

	for j in range(len(labels)):

		current_label = labels[j]
		new_label = reg_range[new_label_ordering.index(current_label)]

		labels[j] = new_label

	labels = labels.reshape(image_a.shape)

	return labels

#really want to try this overnight with bigger images
def image_preproc(img_filename, mask = 0, resize = 0.1, size = (400, 400)):

	img = Image.open(img_filename).convert('L')

	#print(img.size)

	img.thumbnail(size, Image.ANTIALIAS)

	#print(img.size)

	img_arr = np.asarray(img)

	#print(img_arr.shape)



	img_arr = sp.misc.imresize(img_arr, resize) / 255.

	# SMOOTH THIS IS NEW

	if not mask:

		#plt.figure()
		#plt.imshow(img_arr, cmap=plt.cm.gray)
		#plt.axis('off')

		#img_arr = sp.ndimage.gaussian_filter(img_arr, 1)

		img_arr = sp.ndimage.filters.median_filter(img_arr,(3,3))

		#plt.figure()
		#plt.imshow(img_arr, cmap=plt.cm.gray)
		#plt.axis('off')

		#plt.show()



	#print(img_arr.shape)

	return img_arr
#img_fn = 'Data/Unlabeled/normal/C5_Frame (48).jpg'

#img_fn = 'Data/Unlabeled/lesion_c8/C8_Frame (61).jpg'
'''
def segment_stats_row_into_list_test(segment_stats_row):

	l_to_return = []

	l_to_return.append(segment_stats_row['segment_id'])
	l_to_return.append(segment_stats_row['centroid_x'])
	l_to_return.append(segment_stats_row['centroid_y'])
	l_to_return.append(segment_stats_row['total_area'])
	l_to_return.append(segment_stats_row['bb_x_dim'])
	l_to_return.append(segment_stats_row['bb_y_dim'])
	l_to_return.append(segment_stats_row['bb_x_div_total_area'])
	l_to_return.append(segment_stats_row['bb_y_div_total_area'])
	l_to_return.append(segment_stats_row['bb_x_div_bb_y'])
	l_to_return.append(segment_stats_row['total_pixel_intensity'])
	l_to_return.append(segment_stats_row['mean_pixel_intensity'])
	l_to_return.append(segment_stats_row['min_pix'])
	l_to_return.append(segment_stats_row['max_pix'])
	l_to_return.append(segment_stats_row['bb_x_div_total_area'])
	l_to_return.append(segment_stats_row['bb_x_div_total_area'])
	l_to_return.append(segment_stats_row['bb_x_div_total_area'])
	l_to_return.append(segment_stats_row['pix_range'])
	l_to_return.append(segment_stats_row['std'])
	l_to_return.append(segment_stats_row['mad'])
	l_to_return.append(segment_stats_row['skew'])
	l_to_return.append(segment_stats_row['kurtosis'])
	l_to_return.append(segment_stats_row['n_borders'])

	#l_to_return.append(segment_stats_row['min_pix'])
	#l_to_return.append(segment_stats_row['min_pix'])

	l_to_return.extend(segment_stats_row['seg_mean_differences'])

	l_to_return.extend(segment_stats_row['seg_mean_diff_if_bordering'])

	l_to_return.append(segment_stats_row['mean_border_diff'])

	l_to_return.append(segment_stats_row['mean_border_abs_diff'])
	
	return l_to_return
'''

def new_segment_stats_row_into_list_test(segment_stats_row):

	l_to_return = []

	l_to_return.append(segment_stats_row['segment_id'])

	#l_to_return.append(segment_stats_row['mask_count'])
	#l_to_return.append(segment_stats_row['frac_masked'])

	l_to_return.append(segment_stats_row['centroid_x'])
	l_to_return.append(segment_stats_row['centroid_y'])
	l_to_return.append(segment_stats_row['total_area'])
	l_to_return.append(segment_stats_row['bb_x_dim'])
	l_to_return.append(segment_stats_row['bb_y_dim'])
	l_to_return.append(segment_stats_row['bb_x_div_total_area'])
	l_to_return.append(segment_stats_row['bb_y_div_total_area'])
	l_to_return.append(segment_stats_row['bb_x_div_bb_y'])
	l_to_return.append(segment_stats_row['total_pixel_intensity'])
	l_to_return.append(segment_stats_row['mean_pixel_intensity'])
	l_to_return.append(segment_stats_row['min_pix'])
	l_to_return.append(segment_stats_row['max_pix'])
	l_to_return.append(segment_stats_row['bb_x_div_total_area'])
	l_to_return.append(segment_stats_row['bb_x_div_total_area'])
	l_to_return.append(segment_stats_row['bb_x_div_total_area'])
	l_to_return.append(segment_stats_row['pix_range'])
	l_to_return.append(segment_stats_row['std'])
	l_to_return.append(segment_stats_row['mad'])
	l_to_return.append(segment_stats_row['skew'])
	l_to_return.append(segment_stats_row['kurtosis'])
	l_to_return.append(segment_stats_row['n_borders'])

	#l_to_return.append(segment_stats_row['min_pix'])
	#l_to_return.append(segment_stats_row['min_pix'])

	l_to_return.extend(segment_stats_row['seg_mean_differences'])

	l_to_return.extend(segment_stats_row['seg_mean_diff_if_bordering'])

	l_to_return.append(segment_stats_row['mean_border_diff'])

	l_to_return.append(segment_stats_row['mean_border_abs_diff'])

	l_to_return.append(segment_stats_row['tot_edge_pix_intensity_edges'])
	l_to_return.append(segment_stats_row['tot_edge_pix_intensity_nonedges'])
	l_to_return.append(segment_stats_row['tot_edge_pix_intensity_both'])
	l_to_return.append(segment_stats_row['mean_edge_pix_intensity_edges'])
	l_to_return.append(segment_stats_row['mean_edge_pix_intensity_nonedges'])
	l_to_return.append(segment_stats_row['mean_edge_pix_intensity_both'])

	l_to_return.append(segment_stats_row['edge_count'])
	
	return l_to_return

def edge_viewer(im_arr):

	dx = ndimage.sobel(im_arr, 0)  # horizontal derivative
	dy = ndimage.sobel(im_arr, 1)  # vertical derivative
	mag = np.hypot(dx, dy)  # magnitude
	mag *= 255.0 / np.max(mag)  # normalize (Q&D)

	return mag


def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def heatmap_into_row_labelled(heatmap, label, thresh = [0.2, 0.3, 0.4, 0.5, 0.6]):

	l_to_return = []

	flat_heatmap = heatmap.flatten()

	l_to_return.append(label)

	l_to_return.append(np.sum(flat_heatmap))
	l_to_return.append(np.mean(flat_heatmap))
	l_to_return.append(np.median(flat_heatmap))
	l_to_return.append(np.amax(flat_heatmap))
	l_to_return.append(np.amin(flat_heatmap))
	l_to_return.append(np.amax(flat_heatmap) - np.amin(flat_heatmap))

	l_to_return.append(np.percentile(flat_heatmap, 50))
	l_to_return.append(np.percentile(flat_heatmap, 75))
	l_to_return.append(np.percentile(flat_heatmap, 85))
	l_to_return.append(np.percentile(flat_heatmap, 90))
	l_to_return.append(np.percentile(flat_heatmap, 95))

	'''
	s['std'] = np.std(cr_m2)
	s['mad'] = mad(cr_m2)
	s['skew'] = sp.stats.skew(z)
	s['kurtosis'] = sp.stats.kurtosis(z)
	'''

	l_to_return.append(np.std(flat_heatmap))
	l_to_return.append(mad(flat_heatmap))
	l_to_return.append(sp.stats.skew(flat_heatmap))
	l_to_return.append(sp.stats.kurtosis(flat_heatmap))

	flat_heatmap.sort()

	z = flat_heatmap[::-1]

	l_to_return.append(np.sum(z[:int(len(z)/2.0)]))
	l_to_return.append(np.sum(z[:int(len(z)/5.0)]))
	l_to_return.append(np.sum(z[:int(len(z)/10.0)]))
	l_to_return.append(np.sum(z[:int(len(z)/20.0)]))

	l_to_return.append(np.sum(flat_heatmap[np.where(flat_heatmap > thresh[0])]))
	l_to_return.append(np.sum(flat_heatmap[np.where(flat_heatmap > thresh[1])]))
	l_to_return.append(np.sum(flat_heatmap[np.where(flat_heatmap > thresh[2])]))
	l_to_return.append(np.sum(flat_heatmap[np.where(flat_heatmap > thresh[3])]))
	l_to_return.append(np.sum(flat_heatmap[np.where(flat_heatmap > thresh[4])]))

	mid_200 = crop_center(heatmap,200,200)
	mid_100 = crop_center(heatmap,100,100)
	mid_50 = crop_center(heatmap,50,50)
	mid_10 = crop_center(heatmap,10,10)

	mid_200 = np.sum(mid_200.flatten())
	mid_100 = np.sum(mid_100.flatten())
	mid_50 = np.sum(mid_50.flatten())
	mid_10 = np.sum(mid_10.flatten())

	l_to_return.append(mid_200)
	l_to_return.append(mid_100)
	l_to_return.append(mid_50)
	l_to_return.append(mid_10)
	

	return l_to_return


with open('regressors_1.pickle', 'rb') as handle:
    regressors = pickle.load(handle)

print('applying...')



#img_fn = 'Data/Classification/Classification/51/C5111 (88).jpg'
#img_fn = 'Data/Unlabeled/lesion_c4/C4_Frame (150).jpg'



save_fn = 'Data/class_test/heatmap_stats_combined_random_large.csv'

normal_images = glob('Data/class_test/normal/*.jpg')
lesion_images = glob('Data/class_test/lesion/*.jpg')


all_images = []

for n in normal_images:

	all_images.append([0, n])

for l in lesion_images:

	all_images.append([1, l])

shuffle(all_images)
#save_prefix = 'Data/Unlabeled/frame_vids/' + img_pre + '/'

#segmentation_images = glob('Data/Unlabeled/Frames/C' + img_pre + '_*.jpg')

#segmentation_images = sorted_nicely(segmentation_images)

#sl = str(len(segmentation_images))
for i in range(len(all_images)): #1400, 

	img_fn = all_images[i][1]

	print(str(i), img_fn)

	img_high_res = Image.open(img_fn).convert('L')
	img_high_res = np.asarray(img_high_res)

	img_ar = image_preproc(img_fn)

	average_heat_map = np.zeros_like(img_high_res)
	average_heat_map = average_heat_map.astype('float64')
	#average_heat_map = np.zeros_like(img_ar)

	for r in regressors:

		segments = r[0]

		#betas = [5, 10] # NEED TO ROLL OVER BETAS as well!

		b = 10 #betas[0]

		regressor = r[1]

		lbls = spectral_segmentation(img_ar, b, segments)

		seg_stats = new_segment_stats_collector_test(img_ar, lbls, segments)

		#print(seg_stats)

		l_to_predict = []

		for s in seg_stats[1:-1]:

			l_to_predict.append(new_segment_stats_row_into_list_test(s))

		l_to_predict = np.asarray(l_to_predict)

		#print(l_to_predict.shape)

		#real_predictions = list(np.clip(regressor.predict(l_to_predict),0,1))
		real_predictions = list(regressor.predict(l_to_predict))

		real_predictions.insert(0,0)
		real_predictions.insert(len(real_predictions),0)

		#print(real_predictions)
		#print(len(real_predictions))

		#for r in real_predictions:
		#	print(float(r))

		z = np.zeros_like(lbls, dtype=np.float)

		for r in range(lbls.shape[0]):

			for c in range(lbls.shape[1]):

				z[r,c] = real_predictions[lbls[r,c]]

				#average_heat_map[r,c] += real_predictions[lbls[r,c]]
				# resize up! and smooth, then add to average heat map

		#print(z[:25,:25])
		#print(z.shape)
		#print(img_high_res.shape)

		old_max = np.amax(z)

		z = sp.misc.imresize(z, img_high_res.shape, interp = 'nearest', mode = 'L') # Watch that third dimension! for a heatmap
		
		z = old_max * z/np.amax(z)

		plt.figure()
		plt.imshow(z, cmap = plt.cm.jet, vmin=0, vmax=1)
		plt.axis('off')
		plt.show()

		z = sp.ndimage.gaussian_filter(z, sigma=5)

		#print(z.shape)
		#print(z[:25,:25])

		#plt.figure()

		#plt.subplot(1,2,1)
		#plt.imshow(img_high_res, cmap=plt.cm.gray)
		#plt.axis('off')

		
		

		#average_heat_map += z
		# COULD ADD them together and append just one, or collect each z as a row

		
		'''
		l_to_write = heatmap_into_row(z, all_images[i][0])

		with open(save_fn, 'a') as myfile:
			wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
			wr.writerow(l_to_write)
		'''
'''
for i in range(len(normal_images)): #1400, 

	img_fn = normal_images[i]

	print(str(i), img_fn)

	img_high_res = Image.open(img_fn).convert('L')
	img_high_res = np.asarray(img_high_res)

	img_ar = image_preproc(img_fn)

	average_heat_map = np.zeros_like(img_high_res)
	average_heat_map = average_heat_map.astype('float64')
	#average_heat_map = np.zeros_like(img_ar)

	for r in regressors:

		segments = r[0]

		betas = [5, 10] # NEED TO ROLL OVER BETAS as well!

		b = betas[0]

		regressor = r[1]

		lbls = spectral_segmentation(img_ar, b, segments)

		seg_stats = new_segment_stats_collector_test(img_ar, lbls, segments)

		#print(seg_stats)

		l_to_predict = []

		for s in seg_stats[1:-1]:

			l_to_predict.append(new_segment_stats_row_into_list_test(s))

		l_to_predict = np.asarray(l_to_predict)

		#print(l_to_predict.shape)

		#real_predictions = list(np.clip(regressor.predict(l_to_predict),0,1))
		real_predictions = list(regressor.predict(l_to_predict))

		real_predictions.insert(0,0)
		real_predictions.insert(len(real_predictions),0)

		#print(real_predictions)
		#print(len(real_predictions))

		#for r in real_predictions:
		#	print(float(r))

		z = np.zeros_like(lbls, dtype=np.float)

		for r in range(lbls.shape[0]):

			for c in range(lbls.shape[1]):

				z[r,c] = real_predictions[lbls[r,c]]

				#average_heat_map[r,c] += real_predictions[lbls[r,c]]
				# resize up! and smooth, then add to average heat map

		#print(z[:25,:25])
		#print(z.shape)
		#print(img_high_res.shape)

		old_max = np.amax(z)

		z = sp.misc.imresize(z, img_high_res.shape, interp = 'nearest', mode = 'L') # Watch that third dimension! for a heatmap
		
		z = old_max * z/np.amax(z)

		z = sp.ndimage.gaussian_filter(z, sigma=5)

		#print(z.shape)
		#print(z[:25,:25])

		#plt.figure()

		#plt.subplot(1,2,1)
		#plt.imshow(img_high_res, cmap=plt.cm.gray)
		#plt.axis('off')

		#plt.subplot(1,2,2)
		#plt.imshow(z, cmap = plt.cm.jet, vmin=0, vmax=1)
		#plt.axis('off')

		

		#average_heat_map += z
		# COULD ADD them together and append just one, or collect each z as a row

		flat_heat = z.flatten()

		l_to_write = flat_heatmap_into_row(flat_heat, 0)

		with open(save_fn, 'a') as myfile:
			wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
			wr.writerow(l_to_write)

	#average_heat_map /= len(regressors)

	#print(average_heat_map.shape)

	#flat_heat = average_heat_map.flatten()
'''


'''
for i in range(len(lesion_images)):  #range(400, 1400): 

	img_fn = lesion_images[i]

	print(str(i), img_fn)

	img_high_res = Image.open(img_fn).convert('L')
	img_high_res = np.asarray(img_high_res)

	img_ar = image_preproc(img_fn)

	average_heat_map = np.zeros_like(img_high_res)
	average_heat_map = average_heat_map.astype('float64')
	#average_heat_map = np.zeros_like(img_ar)

	for r in regressors:

		segments = r[0]

		betas = [5, 10] # NEED TO ROLL OVER BETAS as well!

		b = betas[0]

		regressor = r[1]

		lbls = spectral_segmentation(img_ar, b, segments)

		seg_stats = new_segment_stats_collector_test(img_ar, lbls, segments)

		#print(seg_stats)

		l_to_predict = []

		for s in seg_stats[1:-1]:

			l_to_predict.append(new_segment_stats_row_into_list_test(s))

		l_to_predict = np.asarray(l_to_predict)

		#print(l_to_predict.shape)

		#real_predictions = list(np.clip(regressor.predict(l_to_predict),0,1))
		real_predictions = list(regressor.predict(l_to_predict))

		real_predictions.insert(0,0)
		real_predictions.insert(len(real_predictions),0)

		#print(real_predictions)
		#print(len(real_predictions))

		#for r in real_predictions:
		#	print(float(r))

		z = np.zeros_like(lbls, dtype=np.float)

		for r in range(lbls.shape[0]):

			for c in range(lbls.shape[1]):

				z[r,c] = real_predictions[lbls[r,c]]

				#average_heat_map[r,c] += real_predictions[lbls[r,c]]
				# resize up! and smooth, then add to average heat map

		#print(z[:25,:25])
		#print(z.shape)
		#print(img_high_res.shape)

		old_max = np.amax(z)

		z = sp.misc.imresize(z, img_high_res.shape, interp = 'nearest', mode = 'L') # Watch that third dimension! for a heatmap
		
		z = old_max * z/np.amax(z)

		z = sp.ndimage.gaussian_filter(z, sigma=5)

		#print(z.shape)
		#print(z[:25,:25])

		#plt.figure()

		#plt.subplot(1,2,1)
		#plt.imshow(img_high_res, cmap=plt.cm.gray)
		#plt.axis('off')

		#plt.subplot(1,2,2)
		#plt.imshow(z, cmap = plt.cm.jet, vmin=0, vmax=1)
		#plt.axis('off')

		

		#average_heat_map += z
		# COULD ADD them together and append just one, or collect each z as a row

		flat_heat = z.flatten()

		l_to_write = flat_heatmap_into_row(flat_heat, 1)

		with open(save_fn, 'a') as myfile:
			wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
			wr.writerow(l_to_write)

	#average_heat_map /= len(regressors)

	#print(average_heat_map.shape)

	#flat_heat = average_heat_map.flatten()


'''