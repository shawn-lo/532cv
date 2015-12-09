import numpy as np
from numpy import linalg as LA
from sklearn.neighbors import KDTree
from plyfile import PlyData, PlyElement

class Normals():
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
#        point = data[0]
#        tree = KDTree(data)
#        dist, ind = tree.query(point, k=k_val)
#        M = np.zeros((3,3))
#        l = []
#        for i in range(1, k_val):
#            diff = data[ind[0][i]]-point
#            outer = np.outer(diff, diff)
#            weight = np.exp(-dist[0,i]/(2*sigma))/dist[0,i]
#            M = M + weight*outer
#            l.append(M)
#        w, v = LA.eig(M)
#        print(w,v)
        l = []
        for point in data:
            tree = KDTree(data)
            dist, ind = tree.query(point, k=k_val)
            M = np.zeros((3,3))
            for i in range(1, k_val):
                diff = data[ind[0][i]]-point
                outer = np.outer(diff, diff)
                weight = np.exp(-dist[0,i]/(2*sigma))/dist[0,i]
                M = M + weight*outer
            w, v = LA.eig(M)
            index = np.argmin(w)
            normal = v[index]
            l.append(normal)
        return l

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



if __name__ == '__main__':
    instance = Normals()
    data1 = instance.load_ply('./hw7/apple_1.ply')
    data2 = instance.load_ply('./hw7/banana_1.ply')
    data3 = instance.load_ply('./hw7/lemon_1.ply')
    l1 = instance.form_scatter(data1)
    l2 = instance.form_scatter(data2)
    l3 = instance.form_scatter(data3)
    instance.write_txt(data1, l1, 'apple_1.txt')
    instance.write_txt(data2, l2, 'banana_1.txt')
    instance.write_txt(data3, l3, 'lemon_1.txt')
