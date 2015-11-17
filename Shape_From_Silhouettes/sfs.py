import numpy as np
import scipy
from scipy import ndimage
from PIL import Image
from pylab import *
import os, sys, glob
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

def construct_voxels(p_list, im_list, scale=10, xRange=5, yRange=6, zRange=2.5, h=582, w=780):
    # construct voxels,(x,y,z)
    # all set to 0, means that at first, all inlier
    # if set value to 1, means that outlier, remove.
    xRange = 5
    yRange = 6
    zRange = 2.5
    scale = 10
    voxels = np.zeros((xRange*scale, yRange*scale, zRange*scale))

    xOffset_global = int(xRange*scale/2)
    yOffset_global = int(yRange*scale/2)
    xOffset_image = int(w/2)
    yOffset_image = int(h/2)
    count = 0
    for x in range(0, int(xRange*scale)):
        for y in range(0, int(yRange*scale)):
            for z in range(0, int(zRange*scale)):
                xCoord = x - xOffset_global
                yCoord = y - yOffset_global
                zCoord = z
                global_coord = np.array([[xCoord,yCoord,zCoord,1]])
                # for each (X,Y,Z,1), dot product with p
                for index in range(0,8):
                    # coord becomes (wx, wy, w).T
                    raw_im_coord = np.dot(p_list[index],global_coord.T)
                    #print(raw_im_coord.shape)
                    # transfer to image coordinate
                    im_x = int(raw_im_coord[0]/raw_im_coord[2] + xOffset_image)
                    im_y = int(-raw_im_coord[1]/raw_im_coord[2] + yOffset_image)
                    im_coord = np.array([im_x, im_y])
                    #print(im_coord)
                    # 1, out of boundary, remove.
                    if im_x < 0 or im_x >= w or im_y < 0 or im_y >= h:
                        voxels[x][y][z] = 1
                    else:
                        count += 1
                    # 2, outlier, remove
#                    else:
#                        im = im_list[index]
#                        if im[im_y][im_x] == 0:
#                            voxels[x][y][z] = 1

#    count = 0
#    for x in range(0, int(xRange*scale)):
#        for y in range(0, int(yRange*scale)):
#            for z in range(0, int(zRange*scale)):
#                if voxels[x][y][z] == 0:
#                    count += 1
    print(count)


if __name__ == '__main__':
    # read from XML files and stroed into P
    p_list = xml2P()
    #print(p)

    # load silhouettes to image list
    im_list = pbm2Im()
    print(im_list[0][200])

    # generate png, nothing
#    idx = 0
#    for im in im_list:
#        print(im)
#        scipy.misc.imsave(str(idx).join(['./ref/','.png']), im)
#        idx += 1

    # construct voxels,(x,y,z)
    construct_voxels(p_list, im_list)
