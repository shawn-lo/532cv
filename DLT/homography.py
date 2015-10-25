from numpy import *
from scipy import ndimage

def DLT(fp, tp):
	if fp.shape != tp.shape:
		raise RuntimeError('The number of from-points and target-points don\'t match.')

	# construct matrix A, 4 points, each point has 2 equation
	A = zeros((8, 9))
	for i in range(4):
		A[2*i] = [fp[0][i], fp[1][i], 1,
				0, 0, 0,
				-fp[0][i]*tp[0][i], -fp[1][i]*tp[0][i], -tp[0][i]]
		A[2*i+1] = [0, 0, 0,
				fp[0][i], fp[1][i], 1,
				-fp[0][i]*tp[1][i], -fp[1][i]*tp[1][i], -tp[1][i]]

	U,E,V = linalg.svd(A)
	H = V[-1].reshape((3,3))
	return H/H[2,2]

def DLT_reverse(fp, tp):
	if fp.shape != tp.shape:
		raise RuntimeError('The number of from-points and target-points don\'t match.')
	
	# construct matrix A, 4 points, each point has 2 equation
	A = zeros((8, 9))
	for i in range(4):
		A[2*i] = [tp[0][i], tp[1][i], 1,
				0, 0, 0,
				-tp[0][i]*fp[0][i], -tp[1][i]*fp[0][i], -fp[0][i]]
		A[2*i+1] = [0, 0, 0,
				tp[0][i], tp[1][i], 1,
				-tp[0][i]*fp[1][i], -tp[1][i]*fp[1][i], -fp[1][i]]
	U,E,V = linalg.svd(A)
	H = V[-1].reshape((3,3))
	#print(H)
	return H/H[2,2]
'''
fp = array([[0, 330, 330, 0], [0, 0, 512, 512], [1,1,1,1]])
tp = array([[341, 421, 421, 341],[24, 24, 140, 140],[1,1,1,1]])
H1 = DLT(fp, tp)
H2 = DLT_reverse(fp, tp)
invH = linalg.inv(H1)
'''
