import numpy as np
import scipy
from scipy import ndimage
from PIL import Image
from pylab import *

# generate rank transformed images
def generateRankImgs(h, w, imgL, imgR, rankWin=5, win=3):
    rankL = np.zeros((h, w), dtype='int64')
    rankR = np.zeros((h, w), dtype='int64')
    rankImg = [rankL, rankR]
    for j in range(0, h):
        for i in range(0,w):
            for q in range(j-int(rankWin/2),j+int(rankWin/2)+1):
                for p in range(i-int(rankWin/2), i+int(rankWin/2)+1):
                    if (q < 0 or q >= h) or (p < 0 or p >= w):
                        tempL = 0
                        tempR = 0
                    else:
                        tempL = imgL[q, p]
                        tempR = imgR[q, p]
                    if tempL < imgL[j,i]:
                        rankL[j,i] += 1
                    if tempR < imgR[j,i]:
                        rankR[j,i] += 1
    return rankImg

# generate disparity map & PKRN
def generateDispMaps(h, w, rankL, rankR, rankWin=5, win=3):
    disparities = np.zeros((h,w), dtype='int64')
    su = np.zeros((h,w), dtype='int64')
    suMin = 0
# About PKRN
    pkrn = np.zeros((h, w), dtype='float')
    arr = [=`=jedi=0, pkrn, disparities]=`= (shape, *_*dtype=float*_*, order='C') =`=jedi=`=
    c1 = np.zeros((h,w), dtype='float')
    c2 = np.zeros((h,w), dtype='float')
    for j in range(0, h):
        for i in range(0, w):
            suMin = rankWin*rankWin*255
            # initial c1 & c2
            c1[j][i] = rankWin*rankWin*255
            c2[j][i] = c1[j][i]+1
            for d in range(0,64):
                su[j][i] = 0
                for q in range(j-int(win/2),j+int(win/2)+1):
                    for p in range(i-int(win/2), i+int(win/2)+1):
                        if (q<0 or q>=h) or (p<0 or p>=w):
                            pixLeft = 0
                            pixRight = 0
                        else:
                            pixLeft = rankL[q, p]
                            if(p-d<=0):
                                pixRight = 0
                            else:
                                pixRight = rankR[q, p-d]
                        su[j][i] += abs(pixLeft - pixRight)
                if(su[j][i] < suMin):
                    suMin = su[j][i]
                    disparities[j][i] = d

                # c1 is the min, c2 is the second min
                if(su[j][i] < c1[j][i]):
                    cur = c1[j][i]
                    c1[j][i] = su[j][i]
                    c2[j][i] = cur
                else:
                    if(su[j][i] <= c2[j][i]):
                        c2[j][i] = su[j][i]
            pkrn[j][i] = c2[j][i]/c1[j][i]
    return arr

if __name__ == '__main__':
    # read Image
    imgR = np.array(Image.open('./teddy/teddyR.pgm'), dtype='int64')
    imgL = np.array(Image.open('./teddy/teddyL.pgm'), dtype='int64')
    # initial
    h, w = imgL.shape[:2]
    # two size of window
    win_3 = 3
    win_15 = 15
    h = int(h)
    w = int(w)

    # Obtain rank images
    rankArr_3 = generateRankImgs(h, w, imgL, imgR, 5, win_3)
    rankArr_15 = generateRankImgs(h, w, imgL, imgR, 5, win_15)
    rankL_3, rankR_3 = rankArr_3[:]
    rankL_15, rankR_15 = rankArr_3[:]

    # obtain disparity map
    pkrn_map, disp_w3 = generateDispMaps(h, w, rankL_3, rankR_3, 5, win_3)
    disp15Arr = generateDispMaps(h, w, rankL_15, rankR_15, 5, win_15)
    disp_w15 = disp15Arr[1]

    # PKRN
    pkrn_sort = np.sort(np.ravel(pkrn_map))
    middle = pkrn_sort[int(450*375/2)-1]
    # the imsave() function may expand the pixel value to fit [0,255].
    scipy.misc.imsave('teddy_disp_w3.png', disp_w3)
    scipy.misc.imsave('teddy_disp_w15.png', disp_w15)

#============================================================
#
# Calculate Error
#
#============================================================
    count_w3 = 0
    count_w15 = 0
    dispImg = np.array(Image.open('./disp2.jpg'), dtype='int64')
    for j in range(0, h):
        for i in range(0, w):
            dispImg[j][i] = np.rint(dispImg[j][i]/4.0)
            if(abs(disp_w3[j][i]-dispImg[j][i]) > 1):
                count_w3 += 1
            if(abs(disp_w15[j][i]-dispImg[j][i]) > 1):
                count_w15 += 1
    error_rate_w3 = count_w3 / (w*h)
    error_rate_w15 = count_w15 / (w*h)
    print('The error rate(3x3) is ', error_rate_w3)
    print('The error rate(15x15) is ', error_rate_w15)


    count1 = 0
    count2 = 0
    for j in range(0, h):
        for i in range(0, w):
            #print(pkrn_map[j][i])
            if pkrn_map[j][i] <= middle:
                disp_w3[j][i] = 100
            else:
                count1 += 1
                # Dealed with dispImg above already.
                #dispImg[j][i] = np.rint(dispImg[j][i]/4.0)
                if(abs(disp_w3[j][i]-dispImg[j][i])>1):
                    count2 += 1
                    print(disp_w3[j][i])
    print(count1, ' ', count2)
    error_rate = count2/count1
    print(error_rate)
    scipy.misc.imsave('teddy_pkrn.png', disp_w3)

