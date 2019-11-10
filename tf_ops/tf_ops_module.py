'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file tf_ops_module.py

    \brief Python definition of the tensorflow operations.

    \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import tensorflow as tf
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
LIB_PATH = os.path.join(BASE_DIR, 'build')
TDModule=tf.load_op_library(os.path.join(LIB_PATH, 'tf_ops_module.so'))

def find_knn(inPts, inSamples, inStartIndexs, inPackedNeighbors, k):
    return TDModule.knn(inPts, inSamples, inStartIndexs, inPackedNeighbors, k)
tf.NoGradient('Knn')

def random_neighbors(inPts, inFeatures, inSamples, inSampleFeatures, inSamplesBatchIds, inStartIndexs, inPackedNeighbors, 
        aabbMin, aabbMax, pdf, radius, batchSize, scaleInv, useFeatures):
    return TDModule.neighbor_rand_select(inPts, inFeatures, inSamples, inSampleFeatures, inSamplesBatchIds, inStartIndexs, inPackedNeighbors, 
        aabbMin, aabbMax, pdf, radius, batchSize, scaleInv, useFeatures)
tf.NoGradient('NeighborRandSelect')

def point_to_mesh_distance(inPts, inVertexs, inFaces, voxFacesIndexs, voxelIndexs, aabbMin, cellSize):
    return TDModule.point_to_mesh_distance(inPts, inVertexs, inFaces, voxFacesIndexs, voxelIndexs, aabbMin, cellSize)
tf.NoGradient('PointToMeshDistance')

def conv_gauss(inPts, inFeatures, inBatchIds, inPDFs, inSamplePts, neighStartIndexs, packedNeighs, aabbMin, aabbMax,
    batchSize, radius, scaleInv):
    return TDModule.spatial_conv_gauss(inPts, inFeatures, inBatchIds, inPDFs, inSamplePts, neighStartIndexs, packedNeighs, 
        aabbMin, aabbMax, batchSize, radius, scaleInv)
tf.NoGradient('SpatialConvGauss')