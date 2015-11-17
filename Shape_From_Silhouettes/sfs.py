import numpy as np
import scipy
from scipy import ndimage
from PIL import Image
from pylab import *
import os, sys, glob
from plyfile import PlyData, PlyElement
import xml.etree.ElementTree as ET

def xml2P():
    P = []
    for i in range(0, 8):
        path = str(i).join(['./dataset/calibration/cam0', '.xml'])
        tree = ET.parse(path)
        root = tree.getroot()
        p_temp = np.fromstring(root.text, sep=' ')
        p_temp = np.reshape(p_temp, (3,4))
        P.append(p_temp)
    return P

def pbm2Im():
    I = []
    for i in range(0,8):
        path = str(i).join(['./dataset/silh_cam0','_00023_0000008550.pbm'])
        im = Image.open(path)
        pixels = im.getdata()
        n = len(pixels)
        data = np.reshape(pixels, (582, 780))
        I.append(data)
    return I

def write2ply(data):
    vertex = np.array(data, dtype=[('x', 'f4'), ('y', 'f4'),('z','f4'),
        ('red','u1'), ('green', 'u1'),('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write('test.ply')

def identify_surface(voxels, scale, xRange, yRange, zRange):
    surface = np.zeros((xRange*scale, yRange*scale, zRange*scale))
    num = 0
    for x in range(0, int(xRange*scale)):
        for y in range(0, int(yRange*scale)):
            for z in range(0, int(zRange*scale)):
                neighbor = 0
                if x-1 >= 0:
                    neighbor += voxels[x-1][y][z]
                if x+1 < xRange*scale:
                    neighbor += voxels[x+1][y][z]
                if y-1 >= 0 :
                    neighbor += voxels[x][y-1][z]
                if y+1 < yRange*scale:
                    neighbor += voxels[x][y+1][z]
                if z-1 >= 0:
                    neighbor += voxels[x][y][z-1]
                if z+1 < zRange*scale:
                    neighbor += voxels[x][y][z+1]
                #if x-1 >= 0 and x+1 < xRange*scale and y-1 >= 0 and y+1 < yRange*scale and z-1>=0 and z+1 < zRange*scale:
                    #neighbor = voxels[x-1][y][z]+voxels[x][y-1][z]+voxels[x][y][z-1]+voxels[x+1][y][z]+voxels[x][y+1][z]+voxels[x][y][z+1]
                if voxels[x][y][z] == 0 and neighbor > 0:
                    surface[x][y][z] = 2
                    num += 1
    print('The num is:',num)
    return surface


def construct_voxels(p_list, im_list, scale=20, xRange=5, yRange=6, zRange=2.5, h=582, w=780):
    # construct voxels,(x,y,z)
    # all set to 0, means that at first, all inlier
    # if set value to 1, means that outlier, remove.
    voxels = np.zeros((xRange*scale, yRange*scale, zRange*scale))
    #omegas = np.zeros((xRange*scale, yRange*scale, zRange*scale))

    xOffset_global = int(xRange*scale/2)
    yOffset_global = int(yRange*scale/2)
    xOffset_image = int(w/2)
    yOffset_image = int(h/2)
    for index in range(0,8):
        count = 0
        for x in range(0, int(xRange*scale)):
            for y in range(0, int(yRange*scale)):
                for z in range(0, int(zRange*scale)):
                    xCoord = (x - xOffset_global)/scale
                    yCoord = (y - yOffset_global)/scale
                    zCoord = z/scale
                    global_coord = np.array([[xCoord,yCoord,zCoord,1]])
                    # for each (X,Y,Z,1), dot product with p
                    # coord becomes (wx, wy, w).T
                    raw_im_coord = np.dot(p_list[index],global_coord.T)
                    #print(raw_im_coord.shape)
                    # transfer to image coordinate
                    omega = raw_im_coord[2]
                    im_x = int(raw_im_coord[0]/omega)
                    im_y = int(raw_im_coord[1]/omega)
                    #im_x = int(raw_im_coord[0]/raw_im_coord[2] + xOffset_image)
                    #im_y = int(-raw_im_coord[1]/raw_im_coord[2] + yOffset_image)
                    im_coord = np.array([im_x, im_y])
                    #print(im_coord)
                    # 1, out of boundary, remove.
                    if im_x < 0 or im_x >= w or im_y < 0 or im_y >= h:
                        voxels[x][y][z] = 1
                    else:
                        #count += 1
                    # 2, outlier, remove
                        im = im_list[index]
                        if im[im_y][im_x] == 0:
                            voxels[x][y][z] = 1
                        else:
                            count += 1

        print(count)

    # identify surface points
    surface = identify_surface(voxels, scale, xRange, yRange, zRange)

    # return a list
    result_list = []
    for x in range(0, int(xRange*scale)):
        for y in range(0, int(yRange*scale)):
            for z in range(0, int(zRange*scale)):
                if surface[x][y][z] != 0:
                    voxels[x][y][z] = 2
                if voxels[x][y][z] == 0:
                    result_list.append((x,y,z,255,0,0))
                if voxels[x][y][z] == 2:
                    result_list.append((x,y,z,255,255,255))
    #result = np.asarray(result_list)
    print(len(result_list))
    return result_list

if __name__ == '__main__':
    # read from XML files and stroed into P
    p_list = xml2P()
    #print(p)

    # load silhouettes to image list
    im_list = pbm2Im()

    # generate png, nothing
#    idx = 0
#    for im in im_list:
#        print(im)
#        scipy.misc.imsave(str(idx).join(['./ref/','.png']), im)
#        idx += 1

    # construct voxels,(x,y,z)
    data = construct_voxels(p_list, im_list)

    # write ply
    write2ply(data)

