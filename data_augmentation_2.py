from PIL import Image
import numpy as np 
import numpy.ma as ma
from numpy import array, argwhere
import matplotlib.pyplot as plt
import scipy as sp 
from scipy import misc
import random
from skimage import transform as tf

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

			else:

				#p = (np.absolute((2*j/les_height) - 1) + np.absolute((2*i/les_width) - 1))/2.0

				half_diag = np.sqrt(les_height**2 + les_width**2)

				dist_from_center = np.sqrt((i - les_width/2)**2 + (j - les_height/2)**2)

				perc_les = float(dist_from_center/half_diag)
				perc_norm = 1 - perc_les

				merged[paste_top_left_y + j, paste_top_left_x + i] = (perc_les * lesion_img[j,i]) + (perc_norm * norm[paste_top_left_y + j, paste_top_left_x + i])

			#merged[paste_top_left_y + j-r:paste_top_left_y + j+r,paste_top_left_x + i-r:paste_top_left_x + i+r] = sp.ndimage.filters.gaussian_filter(merged[paste_top_left_y + j-r:paste_top_left_y + j+r,paste_top_left_x + i-r:paste_top_left_x + i+r], 50)

	merged[paste_top_left_y-r:paste_top_left_y+2*r,paste_top_left_x-r:paste_top_left_x+2*r] = sp.ndimage.filters.gaussian_filter(merged[paste_top_left_y-r:paste_top_left_y+2*r,paste_top_left_x-r:paste_top_left_x+2*r],2)
	merged[paste_top_left_y-r:paste_top_left_y+r,paste_top_left_x-r:paste_top_left_x+r] = sp.ndimage.filters.gaussian_filter(merged[paste_top_left_y-r:paste_top_left_y+r,paste_top_left_x-r:paste_top_left_x+r],2)
	merged[paste_top_left_y+les_height-2*r:paste_top_left_y+les_height+r,paste_top_left_x-r:paste_top_left_x+2*r] = sp.ndimage.filters.gaussian_filter(merged[paste_top_left_y+les_height-2*r:paste_top_left_y+les_height+r,paste_top_left_x-r:paste_top_left_x+2*r],2)




	return merged, new_mask

image_fn = 'Data/Segmentation/C20_image (4).jpg'
mask_fn = 'Data/Segmentation/masks/M20_image (4).jpg'
norm_img_fn = 'Data/Unlabeled/normal/C4_Frame (1).jpg'


img = misc.imread(image_fn, mode = "RGB")
mask = misc.imread(mask_fn)
norm = misc.imread(norm_img_fn, mode = "RGB")

#print(norm.shape)
#print(norm.shape[1])
#print(mask.shape)

img = np.lib.pad(img, ((50, 50), (50, 50), (0, 0)), 'constant', constant_values = 0)
mask = np.lib.pad(mask, ((50,50), (50,50)), 'constant', constant_values = 0)

#print(img.shape)
#print(mask.shape)

img, mask = augment_both(img, mask)

img, mask = smart_crop(img, mask)

merge, merge_mask = paste_lesion_gen_mask(img, mask, norm)

#print(img.shape)
#print(mask.shape)

'''
plt.figure()
plt.imshow(img, cmap=plt.cm.gray)
plt.axis('off')

plt.figure()
plt.imshow(mask, cmap=plt.cm.gray)
plt.axis('off')

plt.figure()
plt.imshow(norm, cmap=plt.cm.gray)
plt.axis('off')
'''

plt.figure()
plt.imshow(merge, cmap=plt.cm.gray)
plt.axis('off')

plt.figure()
plt.imshow(merge_mask, cmap=plt.cm.gray)
plt.axis('off')


plt.show()

'''



mask_mask = mask > 0

coords = np.argwhere(mask_mask)

x0, y0 = coords.min(axis=0)
x1, y1 = coords.max(axis=0) + 1 

img = img[x0:x1, y0:y1]
mask = mask[x0:x1, y0:y1]

#for i in range(img.shape[0]):

#	for j in range(img.shape[1]):

#		img[i,j,3] = mask[i,j]

print(img.shape)
print(mask.shape)
print(norm.shape)

img = image_augment(img)

print(img.shape)

'''

# distort the img and the mask
# select a random position on the norm image
# fill in the surrounding image with faded split %
# place it on the norm image







'''
print(norm.shape)
#print(norm[:5,:5,:])

#img_mask = img[:,:,3] > 0

#coords = np.argwhere(img_mask)

#x0, y0 = coords.min(axis=0)
#x1, y1 = coords.max(axis=0) + 1 

#img = img[x0:x1, y0:y1]
#mask = mask[x0:x1, y0:y1]

#print(img.shape)

center_x = random.randint(120,450)
center_y = random.randint(125,300)

print(center_x,center_y)

for i in range(img.shape[0]):

	for j in range(img.shape[1]):

		if img[i,j,3] > 50:

			norm[center_y + i,  center_x + j,:3] = img[i, j,:3]










img = Image.open(image_fn)#.convert('L')
img_arr = np.asarray(img)

mask = Image.open(mask_fn).convert('L')
mask_arr = np.asarray(mask)

print(img.shape)
img.putalpha(mask)
print(img.shape)

mask.show()
img.show()


masked_arr = np.asarray(img)

print(masked_arr.shape)
'''




#mask_arr = np.clip(mask_arr,0,1)

#pix_masked = ma.array(img_arr, mask = ~mask_arr)
#pix_masked = np.multiply(img_arr, mask_arr)

#print(pix_masked.shape)
'''
B = argwhere(pix_masked)
(ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1 
Atrim = np.array(pix_masked[ystart:ystop, xstart:xstop])

Atrim[Atrim == 0] = 'nan'


new = Image.fromarray(img_arr)
newTrim = Image.fromarray(Atrim)

new.paste(newTrim, (0,0), mask = newTrim)

print(Atrim.shape)

new = np.asarray(new)
'''
#plt.figure()
#plt.imshow(img_arr, cmap=plt.cm.gray)
#plt.axis('off')

#plt.figure()
#plt.imshow(masked_arr, cmap=plt.cm.gray)
#plt.axis('off')

#plt.figure()
#plt.imshow(new, cmap=plt.cm.gray)
#plt.axis('off')






