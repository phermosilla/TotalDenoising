/////////////////////////////////////////////////////////////////////////////
/// \file spatial_conv_gauss.cu
///
/// \brief Cuda implementation of the operations to perform a spatial 
///        convolution on a batch of point clouds.
///
/// \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <iostream>
#include <fstream>

#include "cuda_kernel_utils.h"

#define EXECUTION_BLOCK_SIZE 128

#define SIGMA 0.5657

////////////////////////////////////////////////////////////////////////////////// GPU

/**
 *  Method to compute the weights of the neighbors using a gaussian function.
 *  @param  pScaleInv               Boolean that indicates if the radius is defined relative to the bounding box.
 *  @param  pNumNeighbors           Number of neighboring points.
 *  @param  pNumFeatures            Number of input features per point.
 *  @param  pAABBMin                Minimum point of the bounding box.
 *  @param  pAABBMax                Maximum point of the bounding box.
 *  @param  pSamples                List of samples.
 *  @param  pPoints                 List of points.
 *  @param  pBatchIds               List of batch ids.
 *  @param  pNeigbors               List neighbors of each point.
 *  @param  pPDFs                   List of the pdf values.
 *  @param  pTmpBuff1               Output temp buffer with the weights of each neighbor.
 *  @param  pTmpBuff2               Output temp buffer with the sum of the weights of all neighbors.
 */
__global__ void computeWeightsKernel(
    const bool pScaleInv,
    const int pNumNeighbors,
    const float pRadius,
    const float* __restrict__ pAABBMin,
    const float* __restrict__ pAABBMax,
    const float* __restrict__ pSamples, 
    const float* __restrict__ pPoints, 
    const int* __restrict__ pBatchIds, 
    const int* __restrict__ pNeigbors,
    const float* __restrict__ pPDFs,
    float* __restrict__ pTmpBuffer,
    float* __restrict__ pTmpBuffer2) 
{
    unsigned long long int currentNeighborIndex = threadIdx.x + 
        blockDim.x*(blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y);

    if(currentNeighborIndex < pNumNeighbors){

        int neighborIndex = currentNeighborIndex * 2;
        int currentPointIndex = pNeigbors[neighborIndex];
        int centralPointIndex = pNeigbors[neighborIndex+1];

        int currBatchId = pBatchIds[currentPointIndex];
        float maxAabbSize = max(max(
            pAABBMax[currBatchId*3] - pAABBMin[currBatchId*3], 
            pAABBMax[currBatchId*3+1] - pAABBMin[currBatchId*3+1]), 
            pAABBMax[currBatchId*3+2] - pAABBMin[currBatchId*3+2]);
        float scaledRadius = (pScaleInv)?pRadius*maxAabbSize:pRadius;
        
        float currPointCoords[3] = {
            (pPoints[currentPointIndex*3] - pSamples[centralPointIndex*3])/scaledRadius, 
            (pPoints[currentPointIndex*3+1] - pSamples[centralPointIndex*3+1])/scaledRadius, 
            (pPoints[currentPointIndex*3+2] - pSamples[centralPointIndex*3+2])/scaledRadius};
        float sqrtDist = (currPointCoords[0]*currPointCoords[0] + currPointCoords[1]*currPointCoords[1] +
            currPointCoords[2]*currPointCoords[2]);
        float currPDF = pPDFs[currentNeighborIndex];

        if(sqrtDist > 0.0){
            float invSigma = 1.0/SIGMA;
            float expValue = sqrtDist*0.5*invSigma*invSigma;
            float gaussVal = invSigma*invSigma*invSigma*0.063493636*exp(-expValue);

            pTmpBuffer[currentNeighborIndex] = gaussVal/currPDF;
            atomicAdd(&pTmpBuffer2[centralPointIndex], gaussVal/currPDF);
        }else{
            pTmpBuffer[currentNeighborIndex] = 0.0;
        }
    }
}

/**
 *  Method to evaluate the Gauss kernel.
 *  @param  pNumNeighbors           Number of neighboring points.
 *  @param  pNumFeatures            Number of input features per point.
 *  @param  pFeatures               List of point features.
 *  @param  pNeigbors               List neighbors of each point.
 *  @param  pTmpBuff1               Temp buffer with the weights of each neighbor.
 *  @param  pTmpBuff2               Temp buffer with the sum of the weights of all neighbors.
 *  @param  pFeaturesGrads          Output parameter with the list of convoluted features.
 */
__global__ void evaluateGaussKernel(
    const int pNumNeighbors,
    const int pNumFeatures,
    const float* __restrict__ pFeatures,
    const int* __restrict__ pNeigbors,
    const float* __restrict__ pTmpBuff1,
    const float* __restrict__ pTmpBuff2,
    float* __restrict__ pOutFeatures) 
{
    unsigned long long int currentIndex = threadIdx.x + 
        blockDim.x*(blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y);
    int currentNeighborIndex = currentIndex/pNumFeatures;
    int featureIndex = currentIndex%pNumFeatures;

    if(currentNeighborIndex < pNumNeighbors){

        int neighborIndex = currentNeighborIndex * 2;
        int currentPointIndex = pNeigbors[neighborIndex];
        int centralPointIndex = pNeigbors[neighborIndex+1];

        if(pTmpBuff2[centralPointIndex] > 0.0){
            atomicAdd(&pOutFeatures[centralPointIndex*pNumFeatures+featureIndex], 
                (pFeatures[currentPointIndex*pNumFeatures+featureIndex]*pTmpBuff1[currentNeighborIndex])
                /pTmpBuff2[centralPointIndex]);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////// CPU

void spatialConvGaussCPU(
    bool pScaleInv,
    int pNumNeighbors,
    int pNumInFeatures,
    int pNumSamples,
    float pRadius,
    const float* pInPoints,
    const int* pBatchIds,
    const float* pInFeatures,
    const float* pPDFs,
    const float* pSamples,
    const int* pStartIndexs,
    const int* pPackedNeighs,
    const float* pAABBMin,
    const float* pAABBMax,
    float* pOutFeatues,
    float* pTmpBuff,
    float* pTmpBuff2)
{    
    cudaMemset(pTmpBuff2, 0, sizeof(float)*pNumSamples);
    dim3 gridDimension = computeBlockGrid(
        (unsigned long long int)pNumNeighbors, EXECUTION_BLOCK_SIZE);

    computeWeightsKernel<<<gridDimension, EXECUTION_BLOCK_SIZE>>>(
        pScaleInv, pNumNeighbors, pRadius, pAABBMin, pAABBMax, 
        pSamples, pInPoints, pBatchIds, pPackedNeighs, pPDFs, pTmpBuff, pTmpBuff2);

    cudaMemset(pOutFeatues, 0, pNumInFeatures*pNumSamples*sizeof(float));
    gridDimension = computeBlockGrid(
            (unsigned long long int)pNumNeighbors*
            (unsigned long long int)pNumInFeatures, EXECUTION_BLOCK_SIZE);

    evaluateGaussKernel<<<gridDimension, EXECUTION_BLOCK_SIZE>>>(
        pNumNeighbors, pNumInFeatures, pInFeatures, pPackedNeighs, pTmpBuff, 
        pTmpBuff2, pOutFeatues);

    gpuErrchk(cudaPeekAtLastError());
}