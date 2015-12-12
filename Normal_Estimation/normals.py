import numpy as np
import random
import scipy
from scipy import ndimage
from numpy import linalg as LA
from sklearn.neighbors import KDTree
from plyfile import PlyData, PlyElement

class Normals():
    def normalize(self, vector):
        dim = len(vector)
        length = 0
        for i in range(0,dim):
            length += np.power(vector[i],2)
        length = np.sqrt(length)
        for i in range(0,dim):
            vector[i] = vector[i] / length
        return vector

    def load_ply(self, path):
        plyData = PlyData.read(path)
        data = plyData['vertex']
        vertex_number = len(data[:])
        vertex_data = np.zeros((vertex_number,3),dtype='int64')
        for i in range(0, vertex_number):
            vertex_data[i][0] = data[i][0]
            vertex_data[i][1] = data[i][1]
            vertex_data[i][2] = data[i][2]
        #print(vertex_data)
        return vertex_data

    def form_scatter(self, data, k_val=50, sigma=20):
        k_val = k_val + 1
        l = []
        for point in data:
            tree = KDTree(data)
            dist, ind = tree.query(point, k=k_val)
            M = np.zeros((3,3))
            for i in range(1, k_val):
                index = ind[0][i]
                diff = data[ind[0][i]]-point
                outer = np.outer(diff, diff)
                weight = np.exp(-(dist[0,i]*dist[0,i])/(2*sigma*sigma))/(dist[0,i]*dist[0,i])
                M = M + weight*outer
            w, v = LA.eig(M)
            min_index = np.argmin(w)
            max_index = np.argmax(w)
            #print(min_index)
            #print(max_index)
            eigv = np.array([v[0,min_index],v[1,min_index],v[2,min_index]])
            #eigv2 = np.array([v[0,max_index],v[1,max_index],v[2,max_index]])

            pos = 0
            neg = 0
            for i in range(1, k_val):
                diff = data[ind[0][i]]-point
                if np.dot(diff, eigv) >= 0:
                    pos += 1
                else:
                    neg += 1
            if pos >= neg:
                normal = self.normalize(eigv)
            else:
                normal = self.normalize(eigv*(-1))
            normal = self.normalize(eigv)
            l.append(normal)
        return l

    def spin(self, v, n, p, width=30):
        pv = v - p
        #print(pv)
        pv_len_square = pv[0]*pv[0]+pv[1]*pv[1]
        beta = np.dot(n, pv)
        alpha = np.sqrt(pv_len_square-beta*beta)
        x = alpha
        y = width/2 - beta
        return [x, y]


    def write_txt(self, coords, normals, path):
        size = coords.shape[0]
        data = np.zeros((size, 6))
        for i in range(0, size):
            data[i][0] = coords[i][0]
            data[i][1] = coords[i][1]
            data[i][2] = coords[i][2]
            data[i][3] = normals[i][0]
            data[i][4] = normals[i][1]
            data[i][5] = normals[i][2]
        np.savetxt(path, data, delimiter=',', fmt="%.6f")

    def write_ply(self, coords, normals, path):
        l = []
        size = coords.shape[0]
        for i in range(0, size):
            l.append((coords[i][0], coords[i][1], coords[i][2], normals[i][0], normals[i][1], normals[i][2]))
        vertex = np.array(l, dtype=[('x', 'f4'),('y', 'f4'),('z', 'f4'),('nx','f4'),('ny','f4'),('nz','f4')])
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el]).write(path)


if __name__ == '__main__':
    p1 = Normals()
    data1 = p1.load_ply('./hw7/apple_1.ply')
    data2 = p1.load_ply('./hw7/banana_1.ply')
    data3 = p1.load_ply('./hw7/lemon_1.ply')
    l1 = p1.form_scatter(data1)
    l2 = p1.form_scatter(data2)
    l3 = p1.form_scatter(data3)
    p1.write_txt(data1, l1, 'apple_1.txt')
    p1.write_txt(data2, l2, 'banana_1.txt')
    p1.write_txt(data3, l3, 'lemon_1.txt')

    p1.write_ply(data1, l1, 'apple_1_2.ply')
    p1.write_ply(data2, l2, 'banana_1_2.ply')
    p1.write_ply(data3, l3, 'lemon_1_2.ply')

# Part b and part c
    instance = Normals()
    dataset = []
    for i in range(1,5):
        data = instance.load_ply('./hw7/apple_%s.ply'%i)
        dataset.append(data)
    for i in range(1,5):
        data = instance.load_ply('./hw7/banana_%s.ply'%i)
        dataset.append(data)
    for i in range(1,5):
        data = instance.load_ply('./hw7/lemon_%s.ply'%i)
        dataset.append(data)

    size = len(dataset)
    img_list = []
    spin_accuracy = 0
    cloud_accuracy = 0
    exp_number = 5
    for experiment in range(0,exp_number):
        for i in range(0, size):
            raw_data = dataset[i]
            l = instance.form_scatter(raw_data)
            intensity = 0.85
            number = len(l)
            for j in range(0, 30):
                sImg = np.zeros((11,11))
                indice = random.randint(0,number-1)
                for k in range(0, number):
                    x, y = instance.spin(raw_data[k], l[indice], raw_data[indice])
                    #print(x, y)
                    if 0<= x and x <= 30 and 0 <= y and y <= 30:
                        lu_x_ind = int(x/3)
                        lu_y_ind = int(y/3)
                        a = x/3-int(x/3)
                        b = y/3-int(y/3)
                        # lu
                        sImg[lu_y_ind][lu_x_ind] += (1-a)*(1-b)*intensity
                        # ru
                        if lu_x_ind < 10:
                            sImg[lu_y_ind][lu_x_ind+1] += (1-a)*b*intensity
                        # lb
                        if lu_y_ind < 10:
                            sImg[lu_y_ind+1][lu_x_ind] += a*(1-b)*intensity
                        # rb
                        if lu_x_ind < 10 and lu_y_ind < 10:
                            sImg[lu_y_ind+1][lu_x_ind+1] += a*b*intensity
                file_ind = i*30+j
                #scipy.misc.imsave('./result1/%s.png'%file_ind,sImg)
                img_list.append(sImg)

        # construct data feature and label
        newdata = np.zeros((360,123))
        for i in range(0,360):
            # features
            for y in range(0, 11):
                for x in range(0, 11):
                    feature_ind = y*11+x
                    img = img_list[i]
                    newdata[i][feature_ind] = img[y][x]
            # labels
            # class label
            newdata[i][121] = int(i/120)
            # image label
            newdata[i][122] = int(i/30)

        # split into training and testing
        train_data = np.zeros((180,123))
        test_data = np.zeros((180,123))
        for i in range(0, 6):
            for j in range(0,60):
                if i%2 == 0:
                    for k in range(0, 123):
                        train_data[int(i/2)*60+j][k] = newdata[i*60+j][k]
                else:
                    for k in range(0, 123):
                        test_data[int(i/2)*60+j][k] = newdata[i*60+j][k]

        # test
        correct = 0
        wrong = 0
        img_predicted = np.zeros((6,3))
        for i in range(0,180):
            tree = KDTree(train_data[:,0:121])
            item = test_data[i,0:121]
            dist, ind = tree.query(item, k=1)
            fruit = [0,0,0]
            apple = 0
            banana = 0
            lemon = 0
            for indice in ind[0]:
                print(train_data[indice][121])
                if train_data[indice][121] == 0:
                    fruit[0] += 1
                elif train_data[indice][121] == 1:
                    fruit[1] += 1
                else:
                    fruit[2] += 1
            predicted = np.argmax(fruit)
            img_indice = int(i/30)
            img_predicted[img_indice][predicted] += 1
            if predicted == test_data[i][121]:
                correct += 1
            else:
                wrong += 1
        print(correct, wrong)
        print(correct/(correct+wrong))
        spin_accuracy += correct/(correct+wrong)

        c = 0
        w = 0
        for i in range(0,6):
            vote_arr = img_predicted[i]
            predicted = np.argmax(vote_arr)
            if i < 2:
                if predicted == 0:
                    c+=1
                else:
                    w+=1
            elif 2 <= i and i < 4:
                if predicted == 1:
                    c+=1
                else:
                    w+=1
            else:
                if predicted == 2:
                    c+=1
                else:
                    w+=1
        cloud_accuracy += c/(c+w)
        print('There are %s cloud are right and %s not right'%(c,w))
    spin_accuracy = spin_accuracy/exp_number
    cloud_accuracy = cloud_accuracy/exp_number
    print(spin_accuracy)
    print(cloud_accuracy)

