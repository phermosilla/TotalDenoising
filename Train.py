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
import numpy as np
import pickle
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops'))
MCCNN_DIR = os.path.join(BASE_DIR, 'MCCNN')
sys.path.append(os.path.join(MCCNN_DIR, 'utils'))
sys.path.append(os.path.join(MCCNN_DIR, 'tf_ops'))

from tf_ops_module import find_knn, random_neighbors, point_to_mesh_distance
from PyUtils import visualize_progress
from NoisyDataSet import NoisyDataSet

current_milli_time = lambda: time.time() * 1000.0


def create_loss(positions, predictedPostions, regPoints, lOrderLoss, 
    global_step, totalNumSteps, patchRadius, regTerm, regLambda):

    # Create the main loss.
    diffPos = (positions-predictedPostions)
    diffPos = tf.abs(diffPos)

    if lOrderLoss == 0:
        exponent = 2.0*(1.0 - (tf.to_float(global_step) \
            *tf.constant(1.0/float(totalNumSteps))))
        exponent = tf.maximum(exponent, 1e-8)
        diffPos = diffPos + tf.constant(1e-8)
        regScale = tf.pow(patchRadius, exponent)/(patchRadius*patchRadius)
    elif lOrderLoss == 1:
        exponent = tf.constant(1.0)
        regScale = patchRadius/(patchRadius*patchRadius)
    elif lOrderLoss == 2:
        exponent = tf.constant(2.0)
        regScale = 1.0

    diffPos = tf.pow(diffPos, exponent)

    loss_axes = tf.reduce_sum(diffPos, axis=1)
    loss = tf.reduce_mean(loss_axes)

    # Create the regularization term to avoid clulstering.
    if regTerm:
        regLoss = tf.reduce_mean(tf.reduce_sum(tf.square(regPoints-predictedPostions), axis=1))*regScale
    else:
        regLoss = 0.0

    return loss*(1.0-regLambda)+regLoss*regLambda


def create_trainning(lossGraph, learningRate, maxLearningRate, learningDecayFactor, learningRateDecay, global_step):
    learningRateExp = tf.train.exponential_decay(learningRate, global_step, learningRateDecay, learningDecayFactor, staircase=True)
    learningRateExp = tf.maximum(learningRateExp, maxLearningRate)
    optimizer = tf.train.AdamOptimizer(learning_rate =learningRateExp)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        trainOp = optimizer.minimize(lossGraph)
    return trainOp, learningRateExp


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to train a model.')
    parser.add_argument('--logFolder', default='log', help='Folder of the output models (default: log)')
    
    parser.add_argument('--model', default='MCModel', help='model (default: MCModel)')
    parser.add_argument('--grow', default=64, type=int, help='Grow rate (default: 64)')
    
    parser.add_argument('--numTrainingSteps', default=11250, type=int, help='Maximum number of training steps  (default: 11250)')
    parser.add_argument('--initLearningRate', default=0.005, type=float, help='Init learning rate  (default: 0.005)')
    parser.add_argument('--learningDecayFactor', default=0.7, type=float, help='Learning deacy factor (default: 0.5)')
    parser.add_argument('--learningDecayRate', default=450, type=int, help='Learning decay rate  (default: 450 Steps)')
    parser.add_argument('--maxLearningRate', default=0.00001, type=float, help='Maximum Learning rate (default: 0.00001)')
    
    parser.add_argument('--eval', action='store_true', help='Evaluation during training (default: False)')
    parser.add_argument('--numTrainStepsEval', default=225, type=int, help='Number of training steps before each evaluation  (default: 225)')

    parser.add_argument('--lossOrder', default=0, type=int, help='order of the Lk loss function used (default: 0)')
    parser.add_argument('--regCleanLambda', default=0.1, type=float, help='Regularization lambda value for clean data(default: 0.1)')
    parser.add_argument('--regNoisyLambda', default=0.1, type=float, help='Regularization lambda value for noisy data(default: 0.1)')
    parser.add_argument('--regTerm', action='store_true', help='Regularization term (default: False)')
    parser.add_argument('--scalePrior', default=0.5, type=float, help='Scale used for the prior to select points in the neighbors (default: 0.5)')
    parser.add_argument('--prior', default=1, type=int, help='Prior prob used to select points in the neighbors (0: Uniform, 1: Gaussian, 2: Wendland C2, 3: Inverse multiquadric) (default: 1.0)')
    parser.add_argument('--priorFeatSpace', action='store_true', help='Using prior in feature space too. (default: False)')
    
    parser.add_argument('--dataset', default=0, type=int, help='Dataset (0:Gaussian, 1:ColoredGaussian, 2:Blensor, 3:RueMadame) (default: 0)')
    parser.add_argument('--cleanTargets', action='store_true', help='Use clean models during training (default: false)')
    
    parser.add_argument('--gpu', default='0', help='GPU (default: 0)')
    parser.add_argument('--gpuMem', default=0.75, type=float, help='GPU memory used (default: 0.75)')

    args = parser.parse_args()

    #Create log folder.
    if not os.path.exists(args.logFolder): os.mkdir(args.logFolder)
    os.system('cp %s.py %s' % (args.model, args.logFolder))
    os.system('cp Train.py %s' % (args.logFolder))
    logFile = args.logFolder+"/log.txt"

    #Write execution info.
    with open(logFile, "a") as myFile:
        myFile.write("Model: "+args.model+"\n")
        myFile.write("Grow: "+str(args.grow)+"\n")
        myFile.write("numTrainingSteps: "+str(args.numTrainingSteps)+"\n")
        myFile.write("InitLearningRate: "+str(args.initLearningRate)+"\n")
        myFile.write("learningDecayFactor: "+str(args.learningDecayFactor)+"\n")
        myFile.write("LearningDecayRate: "+str(args.learningDecayRate)+"\n")
        myFile.write("MaxLearningRate: "+str(args.maxLearningRate)+"\n")
        myFile.write("Loss order: "+str(args.lossOrder)+"\n")
        myFile.write("Reg term: "+str(args.regTerm)+"\n")
        myFile.write("regNoisyLambda: "+str(args.regNoisyLambda)+"\n")
        myFile.write("regCleanLambda: "+str(args.regCleanLambda)+"\n")
        myFile.write("scalePrior: "+str(args.scalePrior)+"\n")
        myFile.write("prior: "+str(args.prior)+"\n")
        myFile.write("dataset: "+str(args.dataset)+"\n")
        myFile.write("cleanTargets: "+str(args.cleanTargets)+"\n")
        
    print("Model: "+args.model)
    print("Grow: "+str(args.grow))
    print("numTrainingSteps: "+str(args.numTrainingSteps))
    print("InitLearningRate: "+str(args.initLearningRate))
    print("learningDecayFactor: "+str(args.learningDecayFactor))
    print("LearningDecayRate: "+str(args.learningDecayRate))
    print("MaxLearningRate: "+str(args.maxLearningRate))
    print("Loss order: "+str(args.lossOrder))
    print("Reg term: "+str(args.regTerm))
    print("regNoisyLambda: "+str(args.regNoisyLambda))
    print("regCleanLambda: "+str(args.regCleanLambda))
    print("scalePrior: "+str(args.scalePrior))
    print("prior: "+str(args.prior))
    print("dataset: "+str(args.dataset))
    print("cleanTargets: "+str(args.cleanTargets))
    
    #Load the model
    model = importlib.import_module(args.model)

    #Get train and test datasets        
    mTrainNoisyDataSet = NoisyDataSet(args.dataset, True) 
    if args.eval:
        mTestNoisyDataSet = NoisyDataSet(args.dataset, False) 
    
    print("Noisy: "+str(mTrainNoisyDataSet.modelList_))
    if args.eval:
        print("Test Noisy: "+str(mTestNoisyDataSet.modelList_))
    with open(logFile, "a") as myFile:
        myFile.write("Noisy: "+str(mTrainNoisyDataSet.modelList_)+"\n")

    numUsedModels = mTrainNoisyDataSet.get_num_models()
    numInstances = mTrainNoisyDataSet.get_num_instances()
    print()
    print("##### DATASET")
    print("Used models: "+str(numUsedModels))
    print("Used instances: "+str(numInstances))
    print()

    #Create session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpuMem, visible_device_list=args.gpu)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    #Create variable and place holders
    epoch_step = tf.Variable(0, name='epoch_step', trainable=False)
    inPts = tf.placeholder(tf.float32, [None, 3])
    inPtsShape = tf.shape(inPts)
    inBatchIds = tf.zeros([inPtsShape[0], 1], dtype=tf.int32)
    inFeatures = tf.ones([inPtsShape[0], 1], dtype=tf.float32)
    inFeaturesColor = tf.placeholder(tf.float32, [None, 1])
    isTraining = tf.placeholder(tf.bool, shape=())
    dropVal = tf.placeholder(tf.float32, shape=())
    inPtsClean = tf.placeholder(tf.float32, [None, 3])

    if args.eval:
        inVertexs = tf.placeholder(tf.float32, [None, 3])
        inFaces = tf.placeholder(tf.int32, [None, 3])
        inFaceIndexs  = tf.placeholder(tf.int32, [None])
        inVoxelIndexs = tf.placeholder(tf.int32, [None, None, None, 2])
        inAABBMin = tf.placeholder(tf.float32, [3])
        inCellSizes = tf.placeholder(tf.float32, [1])
        lossEval = tf.placeholder(tf.float32)
        metricsLossTestSummary = tf.summary.scalar('Eval_Loss', lossEval)

    #Increment step operation.
    increment_epoch_step_op = tf.assign(epoch_step, epoch_step+1)

    #Create the network.
    mPointHierarchyIn = model.create_point_hierarchy_input(inPts, inBatchIds, inFeatures, 1, relRad=False)
    mConvBuilder = model.create_convolution_builder(relRad=False, usePDF=True)

    with tf.variable_scope('Denoiser_scope'):
            
        predDisp = model.create_network_parts(mPointHierarchyIn, mConvBuilder, inFeatures, 1, 
            args.grow, isTraining, dropVal)
        predPts = inPts+predDisp
            
        if args.eval:
            mConvBuilderGauss = model.create_convolution_builder(relRad=False, usePDF=False)
            lowFreqDisp = model.create_gaussian_conv(mPointHierarchyIn, predDisp, radius=0.035, relRad = False)
            predEvalDisp = predDisp-lowFreqDisp
            predPtsEval = inPts+predEvalDisp

            distancesGraph, _, _ = point_to_mesh_distance(predPtsEval, 
                inVertexs, inFaces, inFaceIndexs, inVoxelIndexs, inAABBMin, inCellSizes)

    mPointHierarchyPred = model.create_point_hierarchy_output(predPts, inBatchIds, inFeatures, 1, relRad=False)
    mPointHierarchyClean = model.create_point_hierarchy_output(inPtsClean, inBatchIds, inFeatures, 1, relRad=False)    

    #Create losses
    patchRadius = 0.05

    #Loss for clean data.
    if args.cleanTargets:
        neighCleanPts, neighFeatures, _, startIndexsClean, packedNeighsClean = model.create_neighborhood(
            mPointHierarchyClean, mPointHierarchyIn, patchRadius, relRad=False)
        
        knnIndexs = find_knn(neighCleanPts, predPts, startIndexsClean, packedNeighsClean, -1)
        knnIndexsReshaped = tf.reshape(knnIndexs, [-1])
        regCleanPoints = tf.gather(neighCleanPts, knnIndexsReshaped)

        knnRegressIndexs = find_knn(neighCleanPts, predPts, startIndexsClean, packedNeighsClean, 1)
        knnRegressIndexsReshaped = tf.reshape(knnRegressIndexs, [-1])
        regressCleanPoints = tf.gather(neighCleanPts, knnRegressIndexsReshaped)

        diffLoss = create_loss(regressCleanPoints, predPts, regCleanPoints, args.lossOrder, epoch_step, \
            args.numTrainingSteps, patchRadius, args.regTerm, args.regCleanLambda)
    #Loss for noisy data.
    else:
        neighPredPts, _, _, startIndexsPred, packedNeighsPred = model.create_neighborhood(mPointHierarchyPred, 
            mPointHierarchyIn, patchRadius, relRad=False)
        knnIndexs = find_knn(neighPredPts, predPts, startIndexsPred, packedNeighsPred, -1)
        knnIndexsReshaped = tf.reshape(knnIndexs, [-1])
        regPredPoints = tf.gather(neighPredPts, knnIndexsReshaped)

        mPointHierarchyColor = mPointHierarchyIn
        if args.priorFeatSpace:
            mPointHierarchyColor = model.create_point_hierarchy_output(inPts, inBatchIds, inFeaturesColor, 1, relRad=False)

        neighPts, neighFeatures, _, startIndexs, packedNeighs = model.create_neighborhood(mPointHierarchyColor, 
            mPointHierarchyColor, patchRadius, relRad=False)
        randIndexs = random_neighbors(neighPts, neighFeatures, inPts, mPointHierarchyColor.features_[0], inBatchIds, 
            startIndexs, packedNeighs, mPointHierarchyColor.aabbMin_, mPointHierarchyColor.aabbMax_, args.prior, 
            patchRadius*args.scalePrior, 1, False, args.priorFeatSpace)
        randIndexsReshaped = tf.reshape(randIndexs, [-1])
        regressPredPoints =  tf.gather(neighPts, randIndexsReshaped)
        
        diffLoss = create_loss(regressPredPoints, predPts, regPredPoints, args.lossOrder, epoch_step, \
            args.numTrainingSteps, patchRadius, args.regTerm, args.regNoisyLambda)

    #Create the saver
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
        scope='Denoiser_scope'), max_to_keep=10)    

    #Create training
    trainningOp, learningRateExp = create_trainning(
        diffLoss, args.initLearningRate, args.maxLearningRate, 
        args.learningDecayFactor, args.learningDecayRate, epoch_step)

    #Create sumaries
    learningRateSumm = tf.summary.scalar('learninRate', learningRateExp)
    diffLossSummary = tf.summary.scalar('loss_Diff', diffLoss)
    trainingSummary = tf.summary.merge([diffLossSummary, learningRateSumm])

    #Create init variables 
    initVars = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    
    #Create the summary writer
    summary_writer = tf.summary.FileWriter(args.logFolder, sess.graph)
    summary_writer.add_graph(sess.graph)
    
    #Init variables
    sess.run(initVars)
    step = 0
    np.random.seed(int(time.time()))
    
    numEpochs = args.numTrainingSteps//numInstances
    if args.numTrainingSteps%numInstances != 0:
        numEpochs += 1

    minLoss = 10.0
    
    for epoch in range(numEpochs):

        print("")
        print("Epoch: "+str(epoch)+" of "+str(numEpochs))
        with open(logFile, "a") as myfile:
            myfile.write("Epoch: %6d\n" % (epoch))
    
        startEpochTime = current_milli_time()
        startTrainTime = current_milli_time()

        mTrainNoisyDataSet.begin_epoch()
        for epochStep in range(numInstances):
            
            if args.cleanTargets:
                noisyPts, modelName, _ = mTrainNoisyDataSet.get_current_model()
                cleanPts, modelName, _ = mTrainNoisyDataSet.get_current_model(clean=True)

                if args.dataset == 2 or args.dataset == 1:
                    cleanPts = cleanPts[:,0:3]
                noisyPts = noisyPts[:,0:3]

                _, diffLossRes, trainingSummRes = \
                    sess.run([trainningOp, diffLoss, trainingSummary], 
                    {inPts:noisyPts,
                    inPtsClean:cleanPts,
                    isTraining: True,
                    dropVal: 0.8})
                    
                mTrainNoisyDataSet.next()
            else:
                noisyPts, modelName, _ = mTrainNoisyDataSet.get_current_model()
                if args.dataset == 1 or args.dataset == 2:
                    features = noisyPts[:,6:7]/3.0
                else:
                    features = [[1.0] for auxIter, _ in enumerate(noisyPts)]
                noisyPts = noisyPts[:,0:3]

                _, diffLossRes, trainingSummRes = \
                    sess.run([trainningOp, diffLoss, trainingSummary], 
                    {inPts:noisyPts,
                    inFeaturesColor:features, 
                    isTraining: True,
                    dropVal: 0.8})
                    
                mTrainNoisyDataSet.next()
            
            summary_writer.add_summary(trainingSummRes, step)
            endTrainTime = current_milli_time()

            visualize_progress(epochStep, numInstances, "Loss: %.6f | Time: %.4f | %s | Num Pts: %d" % (
                diffLossRes, (endTrainTime-startTrainTime)/1000.0, modelName, len(noisyPts)))

            with open(logFile, "a") as myfile:
                myfile.write("Step: %6d (%4d) | Loss: %.6f | %s | Num Pts: %d\n" % (
                    step, epochStep, diffLossRes, modelName, len(noisyPts)))
            startTrainTime = current_milli_time()

            sess.run(increment_epoch_step_op)
            step += 1

            if args.eval and step%args.numTrainStepsEval ==0:
                modelsError = {}
                mTestNoisyDataSet.begin_epoch()
                for testIter in range(10):
                    initPoints, modelName, modelInstance = mTestNoisyDataSet.get_current_model()

                    voxelization = pickle.load(open("NoisyDataSets/TestMeshes/"+modelName+".vox", "rb"))
                    indexSet = np.array(list(set(voxelization[1].flatten())))
                    auxPt = voxelization[0][indexSet]
                    aabbMinVal = np.amin(auxPt, axis=0)

                    if args.dataset == 1:
                        initPoints = initPoints[:,0:3]

                    minError = 10.0
                    newPoints = initPoints
                    for refIter in range(10):
                        distancesRes, newPoints = \
                                sess.run([distancesGraph, predPtsEval], 
                                {inPts:newPoints,
                                inVertexs: voxelization[0],
                                inFaces: voxelization[1],
                                inFaceIndexs: voxelization[2],
                                inVoxelIndexs: voxelization[3],
                                inAABBMin: aabbMinVal,
                                inCellSizes: [voxelization[5]],
                                isTraining: False,
                                dropVal: 1.0})

                        errorValue = np.mean(distancesRes)

                        if errorValue < minError:
                            minError = errorValue
                        else:
                            break

                    if not(modelInstance in modelsError):
                        modelsError[modelInstance] = [minError]
                    else:
                        modelsError[modelInstance].append(minError)

                    mTestNoisyDataSet.next()

                totalError = 0.0
                for key, value in modelsError.items():
                    currError = np.mean(np.array(value))
                    totalError += currError
                    print("Eval Error ("+str(key)+"): "+str(currError))
                totalError = totalError/float(len(modelsError.keys()))
                print("Eval Error: "+str(totalError))
                if totalError<minLoss:
                    minLoss = totalError
                
                metricsTestSummRes = sess.run(metricsLossTestSummary, {isTraining: False, lossEval: totalError})
                summary_writer.add_summary(metricsTestSummRes, step)

        endEpochTime = current_milli_time() 

        print("Time: "+str((endEpochTime-startEpochTime)/1000.0))
        with open(logFile, "a") as myfile:
            myfile.write("Time: %.6f\n" % ((endEpochTime-startEpochTime)/1000.0))
        print("")

        saver.save(sess, args.logFolder+"/model.ckpt")
