import numpy as np
import scipy
from scipy import ndimage
from PIL import Image
from pylab import *

def non_max_suppression(pixel, x, y, h, w):
    for j in range(y-1, y+2):
        for i in range(x-1, x+2):
            # ignore pixels falls out of the boundaries
            if i >= 0 and i < w and j >= 0 and j < h:
            # if neighbor is larger
                if pixel[j][i] > pixel[y][x]:
                    return False
    return True

def prewitt_convolution(img, axis):
    prewitt = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    if axis == 'y':
        prewitt = np.array([[-1, -1, -1], [0,0,0], [1,1,1]])
    return ndimage.convolve(img,prewitt)

def gaussian_convolution(img):
    gaussian = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])
    return ndimage.convolve(img, gaussian)/273

def get_harris_response(img, sigma=1):
    # first derivative - Prewitt Filter
    imgX = zeros(img.shape)
    imgY = zeros(img.shape)

    imgX = prewitt_convolution(img, 'x')
    imgY = prewitt_convolution(img, 'y')

    # second derivative - Gaussian Smooth
    # use 5x5 kernel
    imgX = gaussian_convolution(imgX)
    imgY = gaussian_convolution(imgY)

    # compute components of the Harris matrix
    Ixx = imgX*imgX
    Iyy = imgY*imgY
    Ixy = imgX*imgY

    #det = Ixx*Iyy-Ixy*Ixy
    det = Ixx*Iyy
    trace = Ixx+Iyy

    #print(det/trace)
    return det/trace

def get_harris_points(response):
    h, w = response.shape
    corner_threshold = 3000
    harris_points = (response > corner_threshold) * 1

    # get coordinates and their values
    coords = np.array(harris_points.nonzero()).T
    print('After gaussian, the points are: ', coords.shape[0])

    # non-maximum suppression
    filtered_coords = []
    for coord in coords:
        if non_max_suppression(response, coord[1], coord[0], h, w ):
            filtered_coords.append(coord)
    candidate_values = [response[coord[0], coord[1]] for coord in coords]
    print('After NMS, the points are: ',len(filtered_coords))
    return filtered_coords

def compute_SAD(coordL, coordR, imgL, imgR, win=3):
    h, w = imgL.shape[:]
    print(h,w)
    sumL=[]
    sumR=[]
    for cL in coordL:
        info = []
        info.append(cL)
        flag = 0
        for j in range(cL[0]-int(win/2), cL[0]+int(win/2)+1):
            for i in range(cL[1]-int(win/2), cL[1]+int(win/2)+1):
                if j >= 0 and j < h:
                    if i >= 0 and i < w:
                        info.append(imgL[j][i])
                    else:
                        flag = 1
                else:
                    flag = 1
        if flag == 0:
            sumL.append(info)

    for cR in coordR:
        info = []
        info.append(cR)
        flag = 0
        for j in range(cR[0]-int(win/2), cR[0]+int(win/2)+1):
            for i in range(cR[1]-int(win/2), cR[1]+int(win/2)+1):
                if j >= 0 and j < h:
                    if i >= 0 and i < w:
                        info.append(imgR[j][i])
                    else:
                        flag = 1
                else:
                    flag = 1
        if flag == 0:
            sumR.append(info)

    dtype = [('sad', float),('ly',float), ('lx',float), ('ry', float), ('rx',float)]
    sad_arr = np.zeros(len(sumL)*len(sumR), dtype=dtype)

    index = 0
    for l in sumL:
        for r in sumR:
            sad = 0
            for i in range(1,10):
                sad += abs(l[i]-r[i])
            # y, x
            sad_arr[index] = (sad, l[0][0], l[0][1], r[0][0], r[0][1])
            index += 1
    sad_arr = np.array(sad_arr, dtype)
    return sad_arr


def cal_correct_wrong(points):
    points = np.sort(points, order=['ly','lx'])
    size = len(points)
    result = []
    i = 0
    while i < size-1:
        sad = points[i][0]
        ry_correspond = points[i][3]
        rx_correspond = points[i][4]
        while points[i][1] == points[i+1][1] and points[i][2] == points[i+1][2] and i < size-2:
            if points[i+1][0] <= sad:
                sad = points[i+1][0]
                ry_correspond = points[i+1][3]
                rx_correspond = points[i+1][4]
            i += 1

        result.append([sad, points[i][1], points[i][2], ry_correspond, rx_correspond])
        i += 1
    correct = len(result)
    wrong = size - correct
    return result, correct, wrong

def get_correct_wrong(dispImg, points):
    points = np.sort(points, order=['ly','lx'])
    size = len(points)
    result = []
    correct = 0
    for p in points:
        if p[1] == p[3]:
            coordX = p[2]
            coordY = p[1]
            disparity = abs(p[2]-p[4])
            if abs(disparity - (dispImg[coordY][coordX])/4) <= 1:
                result.append([disparity, p[1],p[2],p[3],p[4]])
                correct += 1
    wrong = len(points) - correct
    return result, correct, wrong


#=================================================
# Plot images
#=================================================
def appendimages(img1, img2):
    rows1 = img1.shape[0]
    rows2 = img2.shape[0]

    if rows1 < rows2:
        img1 = concatenate((img1, zeros((rows2-rows1, img1.shape[1]))), axis=0)
    elif rows1 > rows2:
        img2 = concatenate((img2, zeros((rows1-rows2, img2.shape[1]))), axis=0)

    return concatenate((img1, img2), axis=1)

def plot_matches(img1, img2, points, show_below=False):
    figure()
    gray()
    img3 = appendimages(img1, img2)
    if show_below:
        img3 = vstack((img3, img3))
    imshow(img3)

    cols1 = img1.shape[1]
    for item in points:
        plot([item[4]+cols1,item[2]], [item[3],item[1]])
    axis('off')
    show()

def plot_harris_points(i, j, img, filtered_coords):
    figure()
    gray()
    imshow(img)
    plot([p[i] for p in filtered_coords],[p[j] for p in filtered_coords], '*')
    axis('off')
    show()

#=================================================
# Main
#=================================================
if __name__ == '__main__':
    imgL = np.array(Image.open('./teddy/teddyL.pgm'), dtype='int64')
    imgR = np.array(Image.open('./teddy/teddyR.pgm'), dtype='int64')

    responseL = get_harris_response(imgL)
    responseR = get_harris_response(imgR)

    coordsL = get_harris_points(responseL)
    coordsR = get_harris_points(responseR)
    # calculate SAD
    sad = compute_SAD(coordsL, coordsR, imgL, imgR)
    # Sort and get top 5%
    sad = np.sort(sad, order='sad')
    percent = 0.05
    index_5 = int(sad.shape[0] * percent)
    top5percent = sad[:index_5]

    dispImg = np.array(Image.open('./teddy/disp2.pgm'), dtype='int64')

    finalR, finalC, finalW = get_correct_wrong(dispImg, top5percent)
    print('Final, the correct is: ', finalC)
    print('Final, the wrong is: ', finalW)
    plot_matches(imgL, imgR, finalR)

    p = 0.1
    while p <= 1.0:
        index = int(sad.shape[0]*p)
        top = sad[:index]
        result, c2, w2 = get_correct_wrong(dispImg, top)
        print('The correct and wrong are: ', c2, w2)
        p += 0.05
        p = round(p, 2)
