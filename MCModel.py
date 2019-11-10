'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCModel.py

    \brief Definition of the network architecture MCDeNoiser2 for weight 
           computation for each point.

    \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import os
import math
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops'))
MCCNN_DIR = os.path.join(BASE_DIR, 'MCCNN')
sys.path.append(os.path.join(MCCNN_DIR, 'utils'))
sys.path.append(os.path.join(MCCNN_DIR, 'tf_ops'))

from MCConvBuilder import PointHierarchy
from MCConvBuilder import ConvolutionBuilder
from MCNetworkUtils import batch_norm_RELU_drop_out, conv_1x1 
from MCConvModule import sort_points_step1, sort_points_step2, find_neighbors, compute_pdf
from tf_ops_module import conv_gauss


def create_neighborhood(pointHierarchyIn, pointHierarchyOut, radius=0.05, relRad = False, inFeatures = None):
    keys, indexs = sort_points_step1(pointHierarchyIn.points_[0], 
        pointHierarchyIn.batchIds_[0], pointHierarchyIn.aabbMin_, 
        pointHierarchyIn.aabbMax_, pointHierarchyIn.batchSize_, 
        radius, relRad)
    if inFeatures is None:
        sortPts, sortBatchs, sortFeatures, cellIndexs = sort_points_step2(
            pointHierarchyIn.points_[0], 
            pointHierarchyIn.batchIds_[0], 
            pointHierarchyIn.features_[0], 
            keys, indexs, 
            pointHierarchyIn.aabbMin_, pointHierarchyIn.aabbMax_, 
            pointHierarchyIn.batchSize_, radius, relRad)
    else:
        sortPts, sortBatchs, sortFeatures, cellIndexs = sort_points_step2(
            pointHierarchyIn.points_[0], 
            pointHierarchyIn.batchIds_[0], 
            inFeatures, 
            keys, indexs, 
            pointHierarchyIn.aabbMin_, pointHierarchyIn.aabbMax_, 
            pointHierarchyIn.batchSize_, radius, relRad)
    startIndexs, packedNeighs = find_neighbors(
            pointHierarchyOut.points_[0], 
            pointHierarchyOut.batchIds_[0], 
            sortPts, cellIndexs, 
            pointHierarchyIn.aabbMin_, 
            pointHierarchyIn.aabbMax_, 
            radius, pointHierarchyIn.batchSize_, relRad)  
    return sortPts, sortFeatures, sortBatchs, startIndexs, packedNeighs


def create_gaussian_conv(pointHierarchyIn, featuresIn, radius=0.05, relRad = False):
    sortPts, sortFeatures, sortBatchs, startIndexs, packedNeighs = create_neighborhood(pointHierarchyIn,
        pointHierarchyIn, radius, relRad, featuresIn)
    pdfs = compute_pdf(sortPts, sortBatchs, pointHierarchyIn.aabbMin_, 
        pointHierarchyIn.aabbMax_, startIndexs, packedNeighs, 0.25, radius, 
        pointHierarchyIn.batchSize_, relRad)
    return conv_gauss(sortPts, sortFeatures, sortBatchs, pdfs, 
        pointHierarchyIn.points_[0], startIndexs, packedNeighs, 
        pointHierarchyIn.aabbMin_, pointHierarchyIn.aabbMax_,
        pointHierarchyIn.batchSize_, radius, relRad)
    

def create_point_hierarchy_input(points, batchIds, features, batchSize, 
    radiusList = [0.025, 0.05], relRad = False, hierarchyName = "MCPtDeNoise_MPH_1"):

    ############################################ Create the point hierarchy
    mPointHierarchy = PointHierarchy(
        inPoints=points, 
        inFeatures=features, 
        inBatchIds=batchIds, 
        radiusList = radiusList, 
        hierarchyName=hierarchyName, 
        batchSize=batchSize,
        relativeRadius=relRad)

    return mPointHierarchy
    
def create_point_hierarchy_output(points, batchIds, features, batchSize, relRad = False):

    ############################################ Create the point hierarchy
    mPointHierarchy = PointHierarchy(
        inPoints=points, 
        inFeatures=features, 
        inBatchIds=batchIds, 
        radiusList = [], 
        hierarchyName="MCPtDeNoise_MPH_2", 
        batchSize=batchSize,
        relativeRadius=relRad)

    return mPointHierarchy

def create_convolution_builder(
    usePDF = True,
    relRad = False):

    ############################################ Create the convolution builder
    return ConvolutionBuilder(
        usePDF = usePDF,
        useAVG = True,
        KDEWindow=0.2,
        relativeRadius = relRad)

def create_network_parts(pointHierarchyIn, convBuilder, features, numInputFeatures, k, 
    isTraining, dropVal, radiusList = [0.05, 0.1]):


    #### Convolution 1
    convFeatures1 = convBuilder.create_convolution(
        convName = "DeNoiser_Conv_1", 
        inPointHierarchy = pointHierarchyIn,
        inPointLevel=0, 
        outPointLevel=1, 
        inFeatures=features, 
        inNumFeatures=numInputFeatures,
        outNumFeatures=k, 
        convRadius=radiusList[0],
        multiFeatureConv=True)

    #### Convolution 2
    bnConvFeatures1 = batch_norm_RELU_drop_out("DeNoiser_Reduce_1_In_BN", convFeatures1, isTraining, True, dropVal)
    bnConvFeatures1 = conv_1x1("DeNoiser_Reduce_1", bnConvFeatures1, k, k*2)
    bnConvFeatures1 = batch_norm_RELU_drop_out("DeNoiser_Reduce_1_Out_BN", bnConvFeatures1, isTraining, True, dropVal)
    convFeatures2 = convBuilder.create_convolution(
        convName="DeNoiser_Conv_2", 
        inPointHierarchy=pointHierarchyIn,
        inPointLevel=1, 
        outPointLevel=2, 
        inFeatures=bnConvFeatures1,
        inNumFeatures=k*2, 
        convRadius=radiusList[1])


    #### Convolution 5
    bnConvFeatures2 = batch_norm_RELU_drop_out("DeNoiser_Reduce_2_In_BN", convFeatures2, isTraining, True, dropVal)
    bnConvFeatures2 = conv_1x1("DeNoiser_Reduce_2", bnConvFeatures2, k*2, k)
    bnConvFeatures2 = batch_norm_RELU_drop_out("DeNoiser_Reduce_2_Out_BN", bnConvFeatures2, isTraining, True, dropVal)
    convFeatures3 = convBuilder.create_convolution(
        convName="DeNoiser_Conv_3", 
        inPointHierarchy=pointHierarchyIn,
        inPointLevel=2, 
        outPointLevel=1, 
        inFeatures=bnConvFeatures2,
        inNumFeatures=k, 
        convRadius=radiusList[1])

    #### Convolution 6
    convFeatures3 = tf.concat([convFeatures3,convFeatures1], axis=1)
    bnConvFeatures3 = batch_norm_RELU_drop_out("DeNoiser_Reduce_3_In_BN", convFeatures3, isTraining, True, dropVal)
    bnConvFeatures3 = conv_1x1("DeNoiser_Reduce_3", bnConvFeatures3, k*2, k)
    bnConvFeatures3 = batch_norm_RELU_drop_out("DeNoiser_Reduce_3_Out_BN", bnConvFeatures3, isTraining, True, dropVal)
    convFeatures4 = convBuilder.create_convolution(
        convName="DeNoiser_Conv_4", 
        inPointHierarchy=pointHierarchyIn,
        inPointLevel=1, 
        outPointLevel=0, 
        inFeatures=bnConvFeatures3,
        inNumFeatures=k, 
        convRadius=radiusList[0],
        multiFeatureConv=True,
        outNumFeatures=3)

    displacements = tf.tanh(convFeatures4)
    if convBuilder.relativeRadius_:
        aabbSizes = tf.norm(pointHierarchyIn.aabbMax_ - pointHierarchyIn.aabbMin_, axis=1)
        ptAABBSizes = tf.tile(tf.reshape(tf.gather(aabbSizes, tf.reshape(pointHierarchyIn.batchIds_[0], [-1])), [-1, 1]), [1, 3])
        displacements = tf.multiply(displacements, ptAABBSizes)

    return displacements*radiusList[0]


