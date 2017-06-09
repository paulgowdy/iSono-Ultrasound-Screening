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

import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

import xgboost
from xgboost import XGBClassifier, XGBRegressor

dataset = []

with open('specseg20_first_full_out.csv', 'rU') as fn:

	reader = csv.reader(fn, quoting=csv.QUOTE_NONNUMERIC)

	for row in reader:
		dataset.append([ float(i) for i in row ])

dataset = np.asarray(dataset)

#print(type(dataset))
print(dataset.shape)
#print(dataset[:2,:])

# split data into X and y
X = dataset[:,2:]
Y = dataset[:,1:2]

print(X.shape)
print(Y.shape)

seed = 40
test_size = 0.01
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

classifier = XGBClassifier()
regressor = XGBRegressor()

regressor.fit(X_train, y_train)

#print(regressor)

y_pred = regressor.predict(X_test)

#print(y_pred)

# evaluate predictions
#accuracy = mean_squared_error(y_test, y_pred)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

'''
for i in range(1000):

	if float(y_test[i]) > 0.7:

		print(y_pred[i], float(y_test[i]))
'''

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

# Big function - does all the work	
def segment_stats_collector_test(img_arr, labels, n_segments):

	'''

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

def spectral_segmentation(image_a, N_REGIONS = 20):

	# Convert the image into a graph with the value of the gradient on the edges
	graph = image.img_to_graph(image_a)

	# Take a decreasing function of the gradient: an exponential
	# The smaller beta is, the more independent the segmentation is of the
	# actual image. For beta=1, the segmentation is close to a voronoi
	beta = 5
	eps = 1e-6
	graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

	# Apply spectral clustering
	reg_range = np.arange(N_REGIONS)

	assign_labels = 'discretize'

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

	labels = labels.reshape(image_a.shape)

	return labels

def image_preproc(img_filename, resize = 0.1):

	img = Image.open(img_filename).convert('L')

	img_arr = np.asarray(img)

	#print(img_arr.shape)

	img_arr = sp.misc.imresize(img_arr, resize) / 255.

	#print(img_arr.shape)

	return img_arr

img_fn = 'Data/Classification/Classification/51/C5110 (58).jpg'

#img_fn = 'Data/Unlabeled/normal/C5_Frame (48).jpg'

#img_fn = 'Data/Unlabeled/lesion_c8/C8_Frame (61).jpg'

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

segments = 20

img_high_res = Image.open(img_fn).convert('L')
img_high_res = np.asarray(img_high_res)

img_ar = image_preproc(img_fn)

lbls = spectral_segmentation(img_ar,segments)

seg_stats = segment_stats_collector_test(img_ar, lbls, segments)

#print(seg_stats)

l_to_predict = []

for s in seg_stats[1:]:

	l_to_predict.append(segment_stats_row_into_list_test(s))

l_to_predict = np.asarray(l_to_predict)

print(l_to_predict.shape)

real_predictions = list(np.clip(regressor.predict(l_to_predict),0,1))


real_predictions.insert(0,0)
#print(real_predictions)

#for r in real_predictions:
#	print(float(r))

z = np.zeros_like(lbls, dtype=np.float)

for r in range(lbls.shape[0]):

	for c in range(lbls.shape[1]):

		z[r,c] = real_predictions[lbls[r,c]]

plt.figure()

plt.subplot(1,2,1)
plt.imshow(img_high_res, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(z, cmap = plt.cm.jet, vmin=0, vmax=0.6)
plt.axis('off')

plt.show()



