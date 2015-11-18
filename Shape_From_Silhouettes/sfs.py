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
                if voxels[x][y][z] == 0 and neighbor > 0:
                    surface[x][y][z] = 2
                    num += 1
    print('The num is:',num)
    return surface


def construct_voxels(p_list, im_list, scale=30, xRange=5, yRange=6, zRange=2.5, h=582, w=780):
    # construct voxels,(x,y,z)
    # all set to 0, means that at first, all inlier
    # if set value to 1, means that outlier, remove.
    voxels = np.zeros((xRange*scale, yRange*scale, zRange*scale))
    nearest_camera = np.zeros((xRange*scale, yRange*scale, zRange*scale))

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
                #if voxels[x][y][z] == 0:
                    #result_list.append((x,y,z,255,0,0))
                if voxels[x][y][z] == 2:
                    result_list.append((x,y,z,255,255,255))
    return result_list

def texture_render(vertex_list, p_list, tex_list, scale=10, xRange=5, yRange=6):
    rgb = []
    nearest_camera = []
    index = 0
    xOffset = int(xRange*scale/2)
    yOffset = int(yRange*scale/2)
    result = []

    for vertex in vertex_list:
        min_depth = 65535
        gx = (vertex[0]-xOffset)/scale
        gy = (vertex[1]-yOffset)/scale
        gz = vertex[2]/scale
        global_coord = np.array([[gx, gy, gz, 1]])
        xCoord = 0
        yCoord = 0
        # Strategy 1, nearest camera
#        camera = 0
#        for i in range(0,8):
#            raw_im_coord = np.dot(p_list[i], global_coord.T)
#            depth = raw_im_coord[2]
#            im_x = int(raw_im_coord[0]/depth)
#            im_y = int(raw_im_coord[1]/depth)
#            if depth < min_depth:
#                min_depth = depth
#                camera = i
#                xCoord = im_x
#                yCoord = im_y
#        rgb = tex_list[camera][yCoord][xCoord]
#        result.append((vertex[0], vertex[1], vertex[2], rgb[0], rgb[1], rgb[2]))

        # Strategy 2, mean of all camera
        r = 0
        g = 0
        b = 0
        for i in range(0,8):
            raw_im_coord = np.dot(p_list[i], global_coord.T)
            depth = raw_im_coord[2]
            im_x = int(raw_im_coord[0]/depth)
            im_y = int(raw_im_coord[1]/depth)

            r += tex_list[i][im_y][im_x][0]
            g += tex_list[i][im_y][im_x][1]
            b += tex_list[i][im_y][im_x][2]
        result.append((vertex[0],vertex[1],vertex[2],int(r/8),int(g/8),int(b/8)))

        # Strategy 3, intensity
#        y = 0
#        min_y = 0
#        camera = 0
#        for i in range(0,8):
#            raw_im_coord = np.dot(p_list[i], global_coord.T)
#            depth = raw_im_coord[2]
#            im_x = int(raw_im_coord[0]/depth)
#            im_y = int(raw_im_coord[1]/depth)
#            y = 0.299*tex_list[i][im_y][im_x][0]+0.587*tex_list[i][im_y][im_x][1]+0.114*tex_list[i][im_y][im_x][2]
#            print(y)
#            if y > min_y:
#                min_y = y
#                camera = i
#                xCoord = im_x
#                yCoord = im_y
#        rgb = tex_list[camera][yCoord][xCoord]
#        result.append((vertex[0], vertex[1], vertex[2], rgb[0], rgb[1], rgb[2]))

    #print(result)
    return result




if __name__ == '__main__':
    # read from XML files and stroed into P
    p_list = xml2P()
    #print(p)

    # load silhouettes to image list
    im_list = pbm2Im()

    # load texture image
    tex_list = []
    for i in range(0,8):
        path = str(i).join(['./dataset/cam0','_00023_0000008550.png'])
        im = np.array(Image.open(path))
        tex_list.append(im)
    # construct voxels,(x,y,z)
    data = construct_voxels(p_list, im_list)

    # RGB render
    result = texture_render(data, p_list, tex_list, 30)
    # write PLY
    write2ply(result)


