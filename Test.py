'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file PtNoise2PtNoise.py

    \brief Code to train a denoiser network.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import math
import time
import argparse
import importlib
import os
from os import listdir
from os.path import isdir, isfile, join
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops'))
MCCNN_DIR = os.path.join(BASE_DIR, 'MCCNN')
sys.path.append(os.path.join(MCCNN_DIR, 'utils'))
sys.path.append(os.path.join(MCCNN_DIR, 'tf_ops'))

from PyUtils import visualize_progress, save_model
from NoisyDataSet import NoisyDataSet
from tf_ops_module import find_knn, point_to_mesh_distance

def tensors_in_checkpoint_file(fileName):
    varlist=[]
    reader = pywrap_tensorflow.NewCheckpointReader(fileName)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        varlist.append(key)
    return varlist

def build_tensors_in_checkpoint_file(loaded_tensors):
    full_var_list = dict()
    for i, tensor_name in enumerate(loaded_tensors):
        try:
            tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name+":0")
            full_var_list[tensor_name] = tensor_aux
        except:
            pass
    return full_var_list

def float_to_color_scale(values, scale = 1.0, color1=np.array([255, 255, 0]), color2=np.array([50, 50, 255])):
    valuesColors = []
    for currVal in values:
        clipVal = min(currVal/scale, 1.0)
        color = color1*clipVal + color2*(1.0-clipVal)
        valuesColors.append([int(color[0]), int(color[1]), int(color[2])])
    return np.array(valuesColors)

current_milli_time = lambda: time.time() * 1000.0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to evaluate PtNoise2PtNoise model.')
    parser.add_argument('--modelsFolder', default='dnTestModels', help='Output folder where to save the denoised point clouds. (default: dnTestModels)')
    parser.add_argument('--inTrainedModel', default='log/model.ckpt', help='Input trained model (default: log/model.ckpt)')
    parser.add_argument('--model', default='MCModel', help='model (default: MCModel)')
    parser.add_argument('--grow', default=64, type=int, help='Grow rate (default: 64)')
    parser.add_argument('--numIters', default=10, type=int, help='Number of iterations (default: 10)')
    parser.add_argument('--numExecs', default=1, type=int, help='Number executions (default: 1)')
    parser.add_argument('--gaussFilter', action='store_true', help='Use gauss filter (default: False)')
    parser.add_argument('--clusterError', action='store_true', help='Use the clustering metric (default: False)')
    parser.add_argument('--saveModels', action='store_true', help='Save models (default: False)')
    parser.add_argument('--noCompError', action='store_true', help='No computation of the error (default: False)')
    parser.add_argument('--histogram', action='store_true', help='Create an histogram of the distances (default: False)')
    parser.add_argument('--dataset', default=0, type=int, help='Dataset (0:Gaussian, 1:ColoredGaussian, 2:Blensor, 3:RueMadame) (default: 0)')
    parser.add_argument('--gpu', default='0', help='GPU (default: 0)')
    parser.add_argument('--gpuMem', default=0.5, type=float, help='GPU memory used (default: 0.5)')

    args = parser.parse_args()

    if args.saveModels:
        if not os.path.exists(args.modelsFolder): os.mkdir(args.modelsFolder)

    print("Models Folder: "+str(args.modelsFolder))
    print("Input trained model: "+str(args.inTrainedModel))
    print("Model: "+args.model)
    print("Grow: "+str(args.grow))
    print("Dataset: "+str(args.dataset))

    #Load the model
    model = importlib.import_module(args.model)

    #Create session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpuMem, visible_device_list=args.gpu)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    #Create variable and place holders
    inPts = tf.placeholder(tf.float32, [None, 3])
    inPtsShape = tf.shape(inPts)
    inBatchIds = tf.zeros([inPtsShape[0], 1], dtype=tf.int32)
    inFeatures = tf.ones([inPtsShape[0], 1], dtype=tf.float32)

    if args.clusterError:
        inPDPts = tf.placeholder(tf.float32, [None, 3])
        inPtsShape2 = tf.shape(inPDPts)
        inPDBatchIds = tf.zeros([inPtsShape2[0], 1], dtype=tf.int32)
        inPDFeatures = tf.ones([inPtsShape2[0], 1], dtype=tf.float32)

    isTraining = tf.placeholder(tf.bool, shape=())
    inVertexs = tf.placeholder(tf.float32, [None, 3])
    inFaces = tf.placeholder(tf.int32, [None, 3])
    inFaceIndexs  = tf.placeholder(tf.int32, [None])
    inVoxelIndexs = tf.placeholder(tf.int32, [None, None, None, 2])
    inAABBMin = tf.placeholder(tf.float32, [3])
    inCellSizes = tf.placeholder(tf.float32, [1])

    #Create the network.
    mPointHierarchyIn = model.create_point_hierarchy_input(inPts, inBatchIds, inFeatures, 1, relRad=False)
    mConvBuilder = model.create_convolution_builder(relRad=False)
    with tf.variable_scope('Denoiser_scope'):
        predDisp = model.create_network_parts(
            pointHierarchyIn=mPointHierarchyIn, 
            convBuilder=mConvBuilder, 
            features=inFeatures, 
            numInputFeatures=1, 
            k=args.grow, 
            isTraining=isTraining, 
            dropVal=1.0)
            
    if args.gaussFilter:
        lowFreqDisp = model.create_gaussian_conv(
            pointHierarchyIn=mPointHierarchyIn, 
            featuresIn = predDisp, 
            radius=0.035)
        predDisp = predDisp-lowFreqDisp
    
    predPts = inPts+predDisp

    distancesGraph, _, _ = point_to_mesh_distance(predPts, 
        inVertexs, inFaces, inFaceIndexs, inVoxelIndexs, inAABBMin, inCellSizes)

    initDistancesGraph, _, _ = point_to_mesh_distance(inPts, 
        inVertexs, inFaces, inFaceIndexs, inVoxelIndexs, inAABBMin, inCellSizes)

    if args.clusterError:
        mPointHierarchyClean = model.create_point_hierarchy_output(inPDPts, inPDBatchIds, inPDFeatures, 1, relRad=False)
        patchRadius = 0.05
        neighPts, _, startIndexs, packedNeighs =  model.create_neighborhood(mPointHierarchyIn, mPointHierarchyClean, patchRadius)
        knnIndexs = find_knn(neighPts, inPDPts, startIndexs, packedNeighs, 1)


    #Create the saver
    varsModelNames = tensors_in_checkpoint_file(args.inTrainedModel)
    varsModel = build_tensors_in_checkpoint_file(varsModelNames)
    print("Loading model: "+args.inTrainedModel)
    saver1 = tf.train.Saver(var_list=varsModel)
    saver1.restore(sess, args.inTrainedModel)

    #Init variables
    step = 0
    epochStep = 0
    np.random.seed(0)#int(time.time()))

    #Look for the test files.
    mTestNoisyDataSet = NoisyDataSet(args.dataset, False, seed=0) 
    print("Noisy: "+str(mTestNoisyDataSet.modelList_))

    #Process the test files
    totalHistogram = np.zeros((20))
    modelsError = {}
    modelsErrorDist = {}
    modelsErrorCluster = {}
    mTestNoisyDataSet.begin_epoch()
    modelIter = 0
    while not(mTestNoisyDataSet.end_epoch()) and modelIter < 200:

        initPoints, modelName, modelInstance = mTestNoisyDataSet.get_current_model()
        batchIds = [[0] for currPt in initPoints]
        features = [[1.0] for currPt in initPoints]

        if not(args.noCompError):
            initPoints = initPoints[:,0:3]
            voxelization = pickle.load(open("NoisyDataSets/TestMeshes/"+modelName+".vox", "rb"))
            indexSet = np.array(list(set(voxelization[1].flatten())))
            auxPt = voxelization[0][indexSet]
            aabbMinVal = np.amin(auxPt, axis=0)

        if args.saveModels and not(args.noCompError):
            distancesRes = \
                sess.run(initDistancesGraph, 
                {inPts:initPoints,
                inBatchIds:batchIds,
                inFeatures:features,
                inVertexs: voxelization[0],
                inFaces: voxelization[1],
                inFaceIndexs: voxelization[2],
                inVoxelIndexs: voxelization[3],
                inAABBMin: aabbMinVal,
                inCellSizes: [voxelization[5]],
                isTraining: False})

        
            distColors = float_to_color_scale(distancesRes, 0.02)
            save_model(args.modelsFolder+"/"+modelName+"_"+modelInstance+"_c", 
                initPoints, distColors)
        elif args.noCompError:
            distColors = [[255, 255, 255] for pt in initPoints]
            save_model(args.modelsFolder+"/"+modelName+"_"+modelInstance+"_c", 
                initPoints, distColors)

        accumErrors = []
        accumErrorsDist = []
        accumErrorsCluster = []
        lastDistances = None
        for execIter in range(args.numExecs):
            minError = 10.0
            minErrorDist = 0.0
            minErrorCluster = 0.0
            newPoints = initPoints
            for refIter in range(args.numIters):
                if not(args.noCompError):
                    newPoints, distancesRes, predDispRes = \
                            sess.run([predPts, distancesGraph, predDisp], 
                            {inPts:newPoints,
                            inBatchIds:batchIds,
                            inFeatures:features,
                            inVertexs: voxelization[0],
                            inFaces: voxelization[1],
                            inFaceIndexs: voxelization[2],
                            inVoxelIndexs: voxelization[3],
                            inAABBMin: aabbMinVal,
                            inCellSizes: [voxelization[5]],
                            isTraining: False})
                    if args.saveModels:
                        distColors = float_to_color_scale(distancesRes, 0.02)
                        save_model(args.modelsFolder+"/"+modelName+"_"+modelInstance+"_"+str(refIter), 
                            newPoints, distColors)

                    distLoss = np.mean(distancesRes)

                    numNansPts = np.sum(np.isnan(newPoints)) 
                    numNansDisp = np.sum(np.isnan(predDispRes))

                    if numNansPts > 0 or numNansDisp > 0:
                        print(numNansPts)
                        print(numNansDisp)

                    clusterLoss = 0.0
                    if args.clusterError:
                        cleanPoints, _, _ = mTestNoisyDataSet.get_current_model(clean=True)
                        if args.dataset == 4:
                            cleanPoints = cleanPoints[:,0:3]
                        knnIndexsRes, neighPtsRes = \
                            sess.run([knnIndexs, neighPts], 
                            {inPts:newPoints,
                            inPDPts:cleanPoints,
                            isTraining: False})

                        clusterDistList = []
                        for ptIter, cleanPt in enumerate(cleanPoints):
                            if knnIndexsRes[ptIter]>=0:
                                currClusterLoss = np.linalg.norm(neighPtsRes[knnIndexsRes[ptIter]]-cleanPt)
                            else:
                                currClusterLoss = patchRadius
                            clusterDistList.append(currClusterLoss)
                            clusterLoss += currClusterLoss
                        clusterLoss = clusterLoss/float(len(cleanPoints))
                        
                        if args.saveModels:
                            distColors = float_to_color_scale(clusterDistList, 0.02, color1=np.array([255, 50, 50]), color2=np.array([50, 255, 50]))
                            save_model(args.modelsFolder+"/"+modelName+"_"+modelInstance+"_"+str(refIter)+"_cluster", 
                                cleanPoints, distColors)
                        
                    errorValue = clusterLoss + distLoss
                    
                    print(errorValue)
                    if errorValue < minError:
                        minError = errorValue
                        minErrorDist = distLoss
                        minErrorCluster = clusterLoss
                    elif not(args.saveModels):
                        break
                    lastDistances = distancesRes
                else:
                    predDispRes, newPoints = \
                            sess.run([predDisp, predPts], 
                            {inPts:newPoints,
                            inBatchIds:batchIds,
                            inFeatures:features,
                            isTraining: False})
                    
                    if args.saveModels:
                        distColors = [[255, 255, 255] for pt in newPoints]
                        save_model(args.modelsFolder+"/"+modelName+"_"+modelInstance+"_"+str(refIter), newPoints, distColors)
            
            if args.histogram:
                currHistogram = np.histogram(lastDistances.flatten(), bins=20)
                totalHistogram = totalHistogram+currHistogram[0]

            visualize_progress(modelIter, mTestNoisyDataSet.get_num_instances()*args.numExecs, 
                modelName+"_"+modelInstance+" Error: "+str(minError))
            modelIter += 1

            accumErrors.append(minError)
            accumErrorsDist.append(minErrorDist)
            accumErrorsCluster.append(minErrorCluster)

        if not(modelInstance in modelsError):
            modelsError[modelInstance] = [np.mean(np.array(accumErrors))]
            modelsErrorDist[modelInstance] = [np.mean(np.array(accumErrorsDist))]
            modelsErrorCluster[modelInstance] = [np.mean(np.array(accumErrorsCluster))]
        else:
            modelsError[modelInstance].append(np.mean(np.array(accumErrors)))
            modelsErrorDist[modelInstance].append(np.mean(np.array(accumErrorsDist)))
            modelsErrorCluster[modelInstance].append(np.mean(np.array(accumErrorsCluster)))

        mTestNoisyDataSet.next()

    totalError = 0.0
    totalErrorDist = 0.0
    totalErrorCluster = 0.0
    print("")
    for key, value in modelsError.items():
        currError = np.mean(np.array(value))
        currErrorDist = np.mean(np.array(modelsErrorDist[key]))
        currErrorCluster = np.mean(np.array(modelsErrorCluster[key]))
        totalError += currError
        totalErrorDist += currErrorDist
        totalErrorCluster += currErrorCluster
        print("Dist: ("+str(key)+"): "+str(currErrorDist))
        print("Cluster: ("+str(key)+"): "+str(currErrorCluster))
        print("Error ("+str(key)+"): "+str(currError))
        print("")
    print("Error Dist: "+str(totalErrorDist/float(len(modelsError.keys()))))
    print("Error Cluster: "+str(totalErrorCluster/float(len(modelsError.keys()))))
    print("Error: "+str(totalError/float(len(modelsError.keys()))))

    print("")
    print(totalHistogram)

    

    