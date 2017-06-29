from PIL import Image
import numpy as np 
import numpy.ma as ma
from numpy import array, argwhere
import matplotlib.pyplot as plt
import scipy as sp 
from scipy import misc
import random

image_fn = 'Data/Segmentation/C4_image (3).jpg'
mask_fn = 'Data/Segmentation/masks/M4_image (3).jpg'


img = misc.imread(image_fn, mode = "RGBA")
print(img.shape)
mask = misc.imread(mask_fn)
print(mask.shape)

mask_mask = mask > 0

coords = np.argwhere(mask_mask)

x0, y0 = coords.min(axis=0)
x1, y1 = coords.max(axis=0) + 1 

img = img[x0:x1, y0:y1]
mask = mask[x0:x1, y0:y1]

print(img.shape)

for i in range(img.shape[0]):

	for j in range(img.shape[1]):

		img[i,j,3] = mask[i,j]

#plt.figure()
#plt.imshow(img, cmap=plt.cm.gray)
#plt.axis('off')



norm_img_fn = 'Data/Unlabeled/normal/C4_Frame (1).jpg'

norm = misc.imread(norm_img_fn, mode = "RGBA")

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








plt.figure()
plt.imshow(norm, cmap=plt.cm.gray)
plt.axis('off')


plt.show()

'''
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






