from PIL import Image
import numpy as np 
import numpy.ma as ma
from numpy import array, argwhere
import matplotlib.pyplot as plt
import scipy as sp 
from scipy import misc
import random
from skimage import transform as tf
import glob

import re 

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def image_augment(img_arr, angle, skew, pixel_noise, res_percent):

	# Create Afine transform
	afine_tf = tf.AffineTransform(shear = skew)

	# Apply transform to image data
	img_arr = tf.warp(img_arr, afine_tf)

	# Rotate
	img = misc.imrotate(img_arr, angle)

	# Pixel noise
	if pixel_noise != 0:

		pix_noise = np.random.randint(-1 * pixel_noise,pixel_noise,(img.shape[0],img.shape[1],3))

		img = img.astype('int64')

		img[:,:,:3] += pix_noise

		img = np.clip(img, 0, 255)

		img = img.astype('uint8')

	img = sp.misc.imresize(img, res_percent)
	
	return img

def augment_both(img, mask):

	rot_angle = random.randint(0,360)
	skew_degree = random.randint(-30,30)/100.0
	resize_percent = random.randint(60,125)
	flip_hor = random.randint(0,1)
	flip_ver = random.randint(0,1)

	#print(rot_angle,skew_degree, resize_percent)

	im = image_augment(img, rot_angle, skew_degree, 17, resize_percent)
	ma = image_augment(mask, rot_angle, skew_degree, 0, resize_percent)

	return im, ma

def smart_crop(img, mask):

	mask_mask = mask > 0

	coords = np.argwhere(mask_mask)

	x0, y0 = coords.min(axis=0)
	x1, y1 = coords.max(axis=0) + 1 

	img = img[x0:x1, y0:y1]
	mask = mask[x0:x1, y0:y1]

	return img, mask

def paste_lesion_gen_mask(lesion_img, mask, norm):

	new_mask = np.zeros_like(norm)

	norm_width, norm_height = norm.shape[1], norm.shape[0]

	norm_center_x, norm_center_y = norm_width/2, norm_height/2

	paste_center_x, paste_center_y = np.random.normal(norm_center_x, norm_center_x/3), np.random.normal(norm_center_y, norm_center_y/3)

	les_width, les_height = lesion_img.shape[1], lesion_img.shape[0]

	les_center_x, les_center_y = les_width/2, les_height/2

	paste_top_left_x, paste_top_left_y = paste_center_x - les_center_x, paste_center_y - les_center_y

	merged = np.copy(norm) 

	r = 30

	for i in range(les_width):

		for j in range(les_height):

			if mask[j,i] > 0:

				new_mask[paste_top_left_y + j, paste_top_left_x + i] = mask[j,i]

				merged[paste_top_left_y + j, paste_top_left_x + i] = lesion_img[j,i]

	merged[paste_top_left_y-15:paste_top_left_y+les_height+15,paste_top_left_x-15:paste_top_left_x+les_width+15] = sp.ndimage.filters.gaussian_filter(merged[paste_top_left_y-15:paste_top_left_y+les_height+15,paste_top_left_x-15:paste_top_left_x+les_width+15],2)
	merged[paste_top_left_y-10:paste_top_left_y+les_height+10,paste_top_left_x-10:paste_top_left_x+les_width+10] = sp.ndimage.filters.gaussian_filter(merged[paste_top_left_y-10:paste_top_left_y+les_height+10,paste_top_left_x-10:paste_top_left_x+les_width+10],2)
	merged[paste_top_left_y-5:paste_top_left_y+les_height+5,paste_top_left_x-5:paste_top_left_x+les_width+5] = sp.ndimage.filters.gaussian_filter(merged[paste_top_left_y-5:paste_top_left_y+les_height+5,paste_top_left_x-5:paste_top_left_x+les_width+5],2)

	return merged, new_mask

#image_fn = 'Data/Segmentation/C20_image (4).jpg'
#mask_fn = 'Data/Segmentation/masks/M20_image (4).jpg'
#norm_img_fn = 'Data/Unlabeled/normal/C4_Frame (1).jpg'

image_files = glob.glob('Data/Segmentation/*.jpg')
mask_files = glob.glob('Data/Segmentation/masks/*.jpg')
norm_files = glob.glob('Data/Unlabeled/normal/*.jpg')

save_folder = 'Data/augmented_data/'

image_files, mask_files = sorted_nicely(image_files), sorted_nicely(mask_files)

for k in range(50):

	try:

		indices = range(len(image_files))

		i = np.random.choice(indices)

		image_fn = image_files[i]
		mask_fn = mask_files[i]
		norm_img_fn = np.random.choice(norm_files)

		img = misc.imread(image_fn, mode = "RGB")
		mask = misc.imread(mask_fn)
		norm = misc.imread(norm_img_fn, mode = "RGB")

		img = np.lib.pad(img, ((50, 50), (50, 50), (0, 0)), 'constant', constant_values = 0)
		mask = np.lib.pad(mask, ((50,50), (50,50)), 'constant', constant_values = 0)

		img, mask = augment_both(img, mask)

		img, mask = smart_crop(img, mask)

		merge, merge_mask = paste_lesion_gen_mask(img, mask, norm)

		sp.misc.imsave(save_folder + 'aug_' + str(k) + '.jpg', merge)
		sp.misc.imsave(save_folder + 'masks/aug_mask_' + str(k) + '.jpg', merge_mask)

	except:

		pass

	'''
	plt.figure()
	plt.imshow(merge, cmap=plt.cm.gray)
	plt.axis('off')

	plt.figure()
	plt.imshow(merge_mask, cmap=plt.cm.gray)
	plt.axis('off')


	plt.show()
	'''
