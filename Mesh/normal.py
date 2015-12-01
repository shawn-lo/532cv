import numpy as np
import scipy
from PIL import Image
from pylab import *
from plyfile import PlyData, PlyElement

# Tranform vector to unit vector
def normalize(vector):
    dim = len(vector)
    length = 0
    for i in range(0,dim):
        length += np.power(vector[i],2)
    length = np.sqrt(length)
    for i in range(0,dim):
        vector[i] = vector[i] / length
    return vector

# Obtain normal vector by using cross product
def get_normal(v1, v2):
    pass

if __name__ == '__main__':
    #1, load PLY data and get vertex array, face array
    plyData = PlyData.read('gargoyle/gargoyle.ply')
    vertex_number = len(plyData['vertex'][:])
    vertex_data = np.zeros((vertex_number,3), dtype='float')
    face_number = len(plyData['face'][:])
    face_data = np.zeros((face_number,3), dtype='uint16')
    for i in range(0, vertex_number):
        vertex_data[i][0] = plyData['vertex'][i][0]
        vertex_data[i][1] = plyData['vertex'][i][1]
        vertex_data[i][2] = plyData['vertex'][i][2]

    for j in range(0, face_number):
        face_data[j] = plyData['face'][j][0]

    #2, calculate cross product
    vertex_normal = np.zeros((vertex_number,3))
    for face in face_data:
        index1 = face[0]
        index2 = face[1]
        index3 = face[2]

        v1 = vertex_data[index1]
        v2 = vertex_data[index2]
        v3 = vertex_data[index3]

        e1 = v2 - v1
        e2 = v3 - v2
        e3 = v1 - v3

        normal1 = np.cross(e3, e1)
        normal2 = np.cross(e1, e2)
        normal3 = np.cross(e2, e3)
        vertex_normal[index1] = np.add(vertex_normal[index1], normal3)
        vertex_normal[index2] = np.add(vertex_normal[index2], normal1)
        vertex_normal[index3] = np.add(vertex_normal[index3], normal2)

    #3, normalize vector
    for i in range(0,vertex_number):
        vertex_normal[i] = normalize(vertex_normal[i])


    #4, write to ply
    result = []
    # The properties are 'x, y, z, nx, ny, nz'
    for i in range(0, vertex_number):
        record = (plyData['vertex'][i][0],plyData['vertex'][i][1],plyData['vertex'][i][2],vertex_normal[i][0],vertex_normal[i][1],vertex_normal[i][2])
        result.append(record)

    #print(result)
    new_vertex_data = np.array(result, dtype=[('x','float32'),('y','float32'),('z','float32'),('nx','float'),('ny','float'),('nz', 'float')])
    elv = PlyElement.describe(new_vertex_data, 'vertex')
    PlyData([elv]).write('p1.ply')

    #5, write to txt
    np.savetxt('test.txt', new_vertex_data, delimiter=' ', fmt="%.6f")
