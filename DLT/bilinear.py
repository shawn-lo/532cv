#from numpy import *
import numpy as np
from scipy import ndimage
from PIL import Image
from pylab import *

def bilinear_interpolate(img, px, py):
	ru = [np.ceil(py), np.ceil(px)]
	rb = [np.floor(py), np.ceil(px)]
	lu = [np.ceil(py), np.floor(px)]
	lb = [np.floor(py), np.floor(px)]

	a = px - lb[1]
	b = py - lb[0]

	R, G, B = (1-a)*(1-b)*img[lb[0], lb[1]]\
			+ a*(1-b)*img[rb[0], rb[1]]\
			+ a*b*img[ru[0], ru[1]]\
			+ (1-a)*b*img[rb[0],rb[1]]


#	print ([R, G, B])
#	result = array([255-R,255-G,255-B], dtype=int)
	result = array([R,G,B], dtype=int)
#	print (result)
	return result


'''The following is test'''
'''
img = array(Image.open('basketball-court.ppm'))
h = 366
w = 488
tImg = zeros((400,500,3))
for x in range(w):
	for y in range(h):
		tp = bilinear_interpolate(img, x, y)
		tImg[y][x] = tp

#bilinear_interpolate(img, 200.24, 100.42)
figure()
imshow(tImg)
axis('equal')
axis('off')
show()
'''

