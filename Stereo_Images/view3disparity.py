import numpy as np
import scipy
from scipy import ndimage
from PIL import Image
from pylab import *

#================================================================
# Construct 3-D points cloud by two disparity map
# @index: the index of view (0 to 5)
# @interval: distance between two view(mm)
# @f: focal length of camera (pixels)
# @b: distance between camera centers (mm)
# @axis: 1, the refIm is left to disIm
#        2, the refIm is right to disIm
#=================================================================
def construct3D(refDisp, tgtDisp, refIndex, tgtIndex, interval=40, f=1247):
    h, w = refDisp.shape[:]
    b = abs(tgtIndex-refIndex)*interval
    refDisp = refDisp/3
    tgtDisp = tgtDisp/3
    px = w/2
    py = h/2

    # coords store point cloud
    coords = []
    # use view0 and view5, get depth for each pixel
    # init z[y][x] to 1, not 0. Because we would divide it later.
    zRef = np.zeros((h,w))
    zTgt = np.zeros((h,w))
    # if disp=0, ignore -> z = 1
    for y in range(0,h):
        for x in range(0,w):
            if refDisp[y][x] != 0:
                zRef[y][x] = b*f/refDisp[y][x]
            else:
                zRef[y][x] = -1
    for y in range(0, h):
        for x in range(0,w):
            if tgtDisp[y][x] != 0:
                zTgt[y][x] = b*f/tgtDisp[y][x]
            else:
                zTgt[y][x] = -1

    # Assume, origin point of global coordinates system is the left-up corner of view0
    # axis X along with the direction from view0 to view5
    # axis Z along with the direction from image to 3-D points.
    # axis Y along with the direction of cross production multipied by X and Z
    # From the reference view, calculate 3-D coordinates.
    # use view0 to generate 3D points cloud
    for y in range(0,h):
        for x in range(0, w):
            if zRef[y][x] > 0:
                x3d = zRef[y][x]*(x-px)/f + refIndex*interval
                y3d = zRef[y][x]*(y-py)/f
                coords.append([x3d, y3d, zRef[y][x]])

    for y in range(0, h):
        for x in range(0, w):
            if zTgt[y][x] > 0:
                x3d = zTgt[y][x]*(x-px)/f + tgtIndex*interval
                y3d = zTgt[y][x]*(y-py)/f
                coords.append([x3d, y3d, zTgt[y][x]])
    #print(len(coords))
    return coords

def project_nearest(coords3d, w=417, h=370, index=3, interval=40, f=1247):
    px = w/2
    py = h/2
    offset = index * interval
    coords = []
    depthMap = np.zeros((h,w), dtype='int64')
    dtype = [('x', int), ('y', int), ('depth', float)]
    size = len(coords3d)
    rawData = np.zeros((size), dtype=dtype)
    i = 0
    for item in coords3d:
        #x = round(f*(item[0]-offset)/item[2] + px)
        #y = round(f*item[1]/item[2] + py)
        x = f*(item[0]-offset)/item[2] + px
        y = f*item[1]/item[2] + py
        depth = item[2]
        rawData[i] = x,y,depth
        i += 1
        #coords.append([x, y, depth])
    #print(rawData)
    # strategy 1: nearest to the camera
    for item in rawData:
        if item[0] >= 0 and item[1] >=0 and item[0] < 417 and item[1] < 370:
            x = item[0]
            y = item[1]
            # neither b or f is 0, so that we don't have point with depth 0.
            if depthMap[y][x] != 0:
                if item[2] < depthMap[y][x]:
                    depthMap[y][x] = item[2]
            else:
                depthMap[y][x] = item[2]
    return depthMap


#============================================================================
def construct_points(disp1, disp2, disp3, interval=40, f=1247):
    h = 370
    w = 417
    b = interval
    px = w/2
    py = h/2

    coords = []
    z1 = np.zeros((h,w))
    z2 = np.zeros((h,w))
    z3 = np.zeros((h,w))

    for y in range(0,h):
        for x in range(0,w):
            if disp1[y][x][0] != 0:
                z1[y][x] = b*f/disp1[y][x][0]
            else:
                z1[y][x] = -1
            if disp2[y][x][0] != 0:
                z2[y][x] = b*f/disp2[y][x][0]
            else:
                z2[y][x] = -1
            if disp3[y][x][0] != 0:
                z3[y][x]= b*f/disp3[y][x][0]
            else:
                z3[y][x] = -1

    for y in range(0, h):
        for x in range(0, w):
            if z1[y][x] > 0:
                x3d = z1[y][x]*(x-px)/f + 1*40
                y3d = z1[y][x]*(y-py)/f
                coords.append([x3d, y3d, z1[y][x], disp1[y][x][1]])

    # for view3, there is no need to 2D->3D->2D
    for y in range(0, h):
        for x in range(0, w):
            if z2[y][x] > 0:
                x3d = z2[y][x]*(x-px)/f + 3*40
                y3d = z2[y][x]*(y-py)/f
                coords.append([x3d, y3d, z2[y][x], disp2[y][x][1]])

    for y in range(0, h):
        for x in range(0, w):
            if z3[y][x] > 0:
                x3d = z3[y][x]*(x-px)/f + 5*40
                y3d = z3[y][x]*(y-py)/f
                coords.append([x3d, y3d, z3[y][x], disp3[y][x][1]])

    return coords

def project_sad(coords3d, w=417, h=370, index=3, interval=40, f=1247):
    px = w/2
    py = h/2
    offset = index * interval
    coords = []
    depthMap = np.zeros((h,w), dtype='int64')
    sadMap = np.zeros((h,w), dtype='int64')
    dtype = [('x', int), ('y', int), ('depth', float), ('sad', int)]
    size = len(coords3d)
    rawData = np.zeros((size), dtype=dtype)
    i = 0
    for item in coords3d:
        x = f*(item[0]-offset)/item[2] + px
        y = f*item[1]/item[2] + py
        depth = item[2]
        sad = item[3]
        rawData[i] = x,y,depth,sad
        i += 1

   # strategy 2: smallest SAD
    for item in rawData:
        if item[0] >= 0 and item[1] >= 0 and item[0] < w and item[1] < h:
            x = item[0]
            y = item[1]
            if sadMap[y][x] != 0:
                if item[3] < sadMap[y][x]:
                    sadMap[y][x] = item[3]
                    depthMap[y][x] = item[2]
            else:
                sadMap[y][x] = item[3]
                depthMap[y][x] = item[2]
    return depthMap

def generateDepthMap(imL, imR, b=40, f=1247, win=3):
    h, w = imL.shape[:]
    disparities = np.zeros((h,w,2), dtype='int64')
    depthMap = np.zeros((h,w), dtype='int64')
    arr = [depthMap, disparities]
    suMin = 0
    su = np.zeros((h,w), dtype='int64')
    for j in range(0, h):
        for i in range(0, w):
            suMin = win*win*255
            for d in range(0, 23):
                #flag = 0
                su[j][i] = 0
                for q in range(j-int(win/2), j+int(win/2)+1):
                    for p in range(i-int(win/2), i+int(win/2)+1):
                        if (q<0 or q>=h) or (p<0 or p >=w):
                            pixLeft = 0
                            pixRight = 0
                        else:
                            pixLeft = imL[q, p]
                            if p-d <= 0:
                                #flag = 1
                                pixRight = 0
                            else:
                                pixRight = imR[q, p-d]
                        su[j][i] += abs(pixLeft - pixRight)
                if su[j][i] > win*win*255:
                    print('error')
#                if flag == 1:
#                    disparities[j][i] = [255, 0]
                if(su[j][i] < suMin):
                    suMin = su[j][i]
                    disparities[j][i] = [d, suMin]
                    if d != 0:
                        depthMap[j][i] = b*f/d
                    else:
                        depthMap[j][i] = 0
    #print(depthMap)
    print(disparities)
    return arr

def plot(img):
    figure()
    gray()
    imshow(img)
    axis('off')
    show()

if __name__ == '__main__':
    im0 = np.array(Image.open('./cloth3/view0.pgm'),dtype='int64')
    im1 = np.array(Image.open('./cloth3/view1.pgm'),dtype='int64')
    im2 = np.array(Image.open('./cloth3/view2.pgm'),dtype='int64')
    im3 = np.array(Image.open('./cloth3/view3.pgm'),dtype='int64')
    im4 = np.array(Image.open('./cloth3/view4.pgm'),dtype='int64')
    im5 = np.array(Image.open('./cloth3/view5.pgm'),dtype='int64')
    im6 = np.array(Image.open('./cloth3/view6.pgm'),dtype='int64')
    disp1 = np.array(Image.open('./cloth3/disp1.pgm'), dtype='int64')
    disp5 = np.array(Image.open('./cloth3/disp5.pgm'), dtype='int64')

    coords3d = construct3D(disp1, disp5, 1, 5)
    depthMap = project_nearest(coords3d)
    scipy.misc.imsave('./ref/disp3.png', depthMap)

    depthMap1, disparityMap1 = generateDepthMap(im1, im2)
#    test = zeros((370, 417))
#    for y in range(0, 370):
#        for x in range(0, 417):
#            test[y][x] = disparityMap1[y][x][0]
#    scipy.misc.imsave('./ref/test.png', test)


    depthMap3, disparityMap3 = generateDepthMap(im3, im4)
    depthMap5, disparityMap5 = generateDepthMap(im5, im6)
    scipy.misc.imsave('./ref/depthMap1.png', depthMap1)
    scipy.misc.imsave('./ref/depthMap3.png', depthMap3)
    scipy.misc.imsave('./ref/depthMap5.png', depthMap5)

    coords = construct_points(disparityMap1, disparityMap3, disparityMap5)
    # adjust to fit using nearest
    newCoords = []
    for item in coords:
        x = item[0]
        y = item[1]
        depth = item[2]
        newCoords.append([x, y, depth])

    finalDepthMap_n = project_nearest(newCoords)
    finalDepthMap_s = project_sad(coords)
    scipy.misc.imsave('./ref/depthMap3_final_n.png', finalDepthMap_n)
    scipy.misc.imsave('./ref/depthMap3_final_s.png', finalDepthMap_s)

#============================

    #calculate error rate
    correct_n = 0
    for y in range(0, 370):
        for x in range(0, 417):
            d1 = 80*1247/depthMap[y][x]
            d2 = 80*1247/finalDepthMap_n[y][x]
            if abs(d1-d2) < 1:
                correct_n += 1
    print('The error rate(nearest) is: ', correct_n/(417*370))

    correct_s = 0
    for y in range(0, 370):
        for x in range(0, 417):
            d1 = 80*1247/depthMap[y][x]
            d2 = 80*1247/finalDepthMap_s[y][x]
            if abs(d1-d2) < 1:
                correct_s += 1
    print('The error rate(SAD) is: ', correct_s/(417*370))
#    scipy.misc.imsave('./ref/view0.png', im0)
#    scipy.misc.imsave('./ref/view1.png', im1)
#    scipy.misc.imsave('./ref/view2.png', im2)
#    scipy.misc.imsave('./ref/view3.png', im3)
#    scipy.misc.imsave('./ref/view4.png', im4)
#    scipy.misc.imsave('./ref/view5.png', im5)
#    scipy.misc.imsave('./ref/view6.png', im6)
#    scipy.misc.imsave('./ref/disp1.png', disp1)
#    scipy.misc.imsave('./ref/disp5.png', disp5)

