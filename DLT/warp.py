from numpy import *
import scipy
from scipy import ndimage
from PIL import Image
from pylab import *

import bilinear as bi
import homography as ht

def warp(fImg, *path):
	h = 500
	w = 940

	tImg = np.full((h,w,3),(0,0,0), dtype=int)
# find relationship, get matrix H.
	fp = array([[0,0,w,w], [0,h,h,0], [1,1,1,1]])
#	tp = array([[404,280,25,248], [76, 280, 194, 51], [1,1,1,1]])
	tp = array([[25,280,404,248], [194, 280, 76, 51], [1,1,1,1]])
	H = linalg.inv(ht.DLT_reverse(fp,tp))

#	tempX = H[0].dot([w,0,1]) / H[2].dot([w,0,1])
#	tempY = H[1].dot([w,0,1]) / H[2].dot([w,0,1])
#	print(tempX, tempY)
	
	for x in range(w):
		for y in range(h):
			vector = array([x,y,1])
			u = H[0].dot(vector) / H[2].dot(vector)
			v = H[1].dot(vector) / H[2].dot(vector)
			if(u>=405 or v >= 281):
				print('error')
			rp = bi.bilinear_interpolate(fImg, u, v)
			tImg[y][x] = rp
	return tImg

img = array(Image.open('basketball-court.ppm'))
resultImg = warp(img)
scipy.misc.imsave('court_above.jpg', resultImg)
print('success!!')
'''
figure()
imshow(resultImg)
axis('equal')
axis('on')
show()
'''
