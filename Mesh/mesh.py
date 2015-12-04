import numpy as np
import scipy
from scipy import ndimage
from PIL import Image
from pylab import *
from plyfile import PlyData, PlyElement

#======================================
# Disparity
#======================================
def construct3D(img_disp, index, scale=4, interval=40, f=1247):
    h, w = img_disp.shape[:]
    b = interval*scale
    img_disp = img_disp/3
    px = w/2
    py = h/2

    z = np.zeros((h,w))
    coords = []
    # if disp = 0, ignor -> z = 1
    for y in range(0, h):
        for x in range(0, w):
            if img_disp[y][x] != 0:
                z[y][x] = b*f/img_disp[y][x]
            else:
                z[y][x] = -1

    for y in range(0,h):
        for x in range(0,w):
            x3d = z[y][x]*(x-px)/f + index*interval
            y3d = z[y][x]*(y-py)/f
            #coords.append([x3d, y3d, z[y][x]])
            coords.append((x3d, y3d, z[y][x]))
    return coords

def construct_and_texture(img_disp, img, index, scale=2, interval=40, f=1247):
    h, w = img_disp.shape[:]
    b = interval*scale
    img_disp = img_disp/3
    px = w/2
    py = h/2

    z = np.zeros((h,w))
    vertex = []
    # if disp = 0, ignor -> z = 1
    for y in range(0, h):
        for x in range(0, w):
            if img_disp[y][x] != 0:
                z[y][x] = b*f/img_disp[y][x]
            else:
                z[y][x] = -1

    for y in range(0,h):
        for x in range(0,w):
            x3d = z[y][x]*(x-px)/f + index*interval
            y3d = z[y][x]*(y-py)/f
            r = img[y][x][0]
            g = img[y][x][1]
            b = img[y][x][2]
            vertex.append((x3d, y3d, z[y][x], r, g, b))
    return vertex



def project_nearest(coords3d, b=80, w=417, h=370, index=3, interval=40, f=1247):
    px = w/2
    py = h/2
    offset = index * interval
    coords = []
    depthMap = np.zeros((h,w), dtype='float64')
    dispMap = np.zeros((h,w), dtype='int64')
    raw_dispMap = np.zeros((h,w),dtype='float64')
    dtype = [('x', int), ('y', int), ('depth', float), ('disp', float)]
    size = len(coords3d)
    rawData = np.zeros((size), dtype=dtype)
    i = 0
    for item in coords3d:
        x = f*(item[0]-offset)/item[2] + px
        y = f*item[1]/item[2] + py
        depth = item[2]
        if depth == -1:
            disp = 0
        else:
            disp = b*f/depth
        rawData[i] = x,y,depth,disp
        i += 1
    # nearest to the camera
    for item in rawData:
        if item[0] >= 0 and item[1] >=0 and item[0] < 417 and item[1] < 370:
            x = item[0]
            y = item[1]
            # neither b or f is 0, so that we don't have point with depth 0.
            if depthMap[y][x] != 0:
                if item[2] < depthMap[y][x]:
                    depthMap[y][x] = item[2]
                    dispMap[y][x] = item[3]
                    raw_dispMap[y][x] = item[3]
            else:
                depthMap[y][x] = item[2]
                dispMap[y][x] = item[3]
                raw_dispMap[y][x] = item[3]
    return raw_dispMap

def generate_mesh(disp_map):
    h, w = disp_map.shape[:]
    face=[]
    # up triangle.
    for y in range(0, h-1):
        for x in range(0, w-1):
            if disp_map[y][x] != 0 and disp_map[y][x+1] != 0 and disp_map[y+1][x+1] != 0:
                # up triangle
                uindex1 = y*w+x
                uindex3 = uindex1+1
                uindex2 = uindex3+w
                face.append(([uindex1, uindex2, uindex3],))
            if disp_map[y][x] != 0 and disp_map[y+1][x] != 0 and disp_map[y+1][x+1] != 0:
                # down triangle
                dindex1 = y*w+x
                dindex2 = dindex1+w
                dindex3 = dindex2+1
                face.append(([dindex1, dindex2, dindex3],))
    return face

if __name__ == '__main__':
    # load image
    img0 = np.array(Image.open('gargoyle/disp1.pgm'), dtype='int64')
    img1 = np.array(Image.open('gargoyle/view3.png'))
    img2 = np.array(Image.open('gargoyle/disp5.pgm'), dtype='int64')

    # construct 3d coordinates
    coords3d = []
    coords3d1 = construct3D(img0, 1)
    coords3d2 = construct3D(img2, 5)
    coords3d = coords3d1 + coords3d2

    # get disp map
#    depth_map = project_nearest(coords3d)
    disp_map = project_nearest(coords3d)
#    scipy.misc.imsave('disp3.png', disp_map)

    #connect
    raw_face = generate_mesh(disp_map)
    raw_vertex = construct_and_texture(disp_map, img1, 3)
    vertex = np.array(raw_vertex, dtype=[('x', 'f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    face = np.array(raw_face, dtype=[('vertex_indices','int32',(3,))])
    elv = PlyElement.describe(vertex, 'vertex')
    elf = PlyElement.describe(face, 'face')
    PlyData([elv,elf]).write('mesh.ply')






