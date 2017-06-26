import imageio
from glob import glob

import re 

def sorted_nicely( l ): 
	""" Sort the given iterable in the way that humans expect.""" 
	convert = lambda text: int(text) if text.isdigit() else text 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(l, key = alphanum_key)

images = []

filenames = glob('bb_videos/c20_bb/*.png')
filenames = sorted_nicely(filenames)

for filename in filenames:
    images.append(imageio.imread(filename))

kargs = {'duration': 0.14}
imageio.mimsave('bb_videos/c20_bb.gif', images, **kargs)