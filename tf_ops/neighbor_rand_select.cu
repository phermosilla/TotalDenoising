/////////////////////////////////////////////////////////////////////////////
/// \file neighbor_rand_select.cu
///
/// \brief CUDA operation definition to select a random point from the 
///     receptive field.
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
#include <time.h>

#include <curand.h>
#include <curand_kernel.h>

#include "cuda_kernel_utils.h"

#define SAMLPES_BLOCK_SIZE 64

////////////////////////////////////////////////////////////////////////////////// GPU

__global__ void compute_rand_pts(
    const bool pUseFeatures,
    const unsigned int pSeed,
    const bool pScaleInv,
    const int pdf,
    const float radius,
    const int pNumSamples,
    const int pNumNeighbors,
    const int pNumFeatures,
    const float* __restrict__ pInPts,
    const float* __restrict__ pInFeatures,
    const float* __restrict__ pInSamples,
    const float* __restrict__ pInSampleFeatures,
    const int*  __restrict__ pBatchIds,
    const int* __restrict__ pStartIndexs,
    const int* __restrict__ pPackedIndexs,
    const float* __restrict__ pAABBMin,
    const float* __restrict__ pAABBMax,
    int* __restrict__ pPtsIndices)
{
    int currentSampleIndex = threadIdx.x + blockDim.x*(blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y);
    if(currentSampleIndex < pNumSamples){
        curandState localState;
        curand_init(pSeed, currentSampleIndex, 0, &localState);

        int initIter = pStartIndexs[currentSampleIndex];
        int endIter = (currentSampleIndex < pNumSamples-1)?pStartIndexs[currentSampleIndex+1]:pNumNeighbors;
        float interval = (float)(endIter-initIter);

        float centerPointCoords[3] = {
            pInSamples[currentSampleIndex*3], 
            pInSamples[currentSampleIndex*3+1], 
            pInSamples[currentSampleIndex*3+2]};
        
        int selectedIndex = -1;

        if(pdf ==0){
            int neighIndex = min((int)ceil(curand_uniform(&localState)*interval + (float)initIter), endIter)-1;
            selectedIndex = pPackedIndexs[neighIndex*2];
        }else{
            int currBatchId = pBatchIds[currentSampleIndex];
            float maxAabbSize = max(max(
                pAABBMax[currBatchId*3] - pAABBMin[currBatchId*3], 
                pAABBMax[currBatchId*3+1] - pAABBMin[currBatchId*3+1]), 
                pAABBMax[currBatchId*3+2] - pAABBMin[currBatchId*3+2]);
            float scaledRadius = (pScaleInv)?radius*maxAabbSize:radius;

            int counter = 0;
            int lastIndex = -1;
            while(interval > 0.0 && selectedIndex < 0 && counter < 150){
                int currNeighIndex = min((int)ceil(curand_uniform(&localState)*interval + (float)initIter), endIter)-1;
                int currIndex = pPackedIndexs[currNeighIndex*2];
                float currPointCoords[3] = {
                    pInPts[currIndex*3], 
                    pInPts[currIndex*3+1], 
                    pInPts[currIndex*3+2]};
                float diff [3] = {
                    (currPointCoords[0] - centerPointCoords[0]), 
                    (currPointCoords[1] - centerPointCoords[1]), 
                    (currPointCoords[2] - centerPointCoords[2])};
                float distance = (diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])/(scaledRadius*scaledRadius);
                
                if(pUseFeatures){
                    float distanceFeatures = 0.0;
                    for(int fIter = 0; fIter < pNumFeatures; ++fIter)
                    {
                        float auxValue = pInFeatures[currIndex*pNumFeatures+fIter] - 
                            pInSampleFeatures[currentSampleIndex*pNumFeatures+fIter];
                        distanceFeatures += auxValue*auxValue;
                    }
                    distance += distanceFeatures*0.5;
                }

                distance = sqrt(distance);

                float prob = 0.0;
                if(pdf ==1){
                    prob = exp(-(distance*distance)*1.5625)*0.705236979; //Gaussian: STD=0.5657
                }else if(pdf == 2){
                    prob = pow(max(1.0f-distance, 0.0f), 4.0f)*(1.0f+4.0f*distance); //Wendland C2
                }else if(pdf == 3){
                    prob = 1.0f/sqrt(1.0f+pow(distance*5.0f, 2.0f)); //Inverse multiquadric with E=5.0
                }

                if(curand_uniform(&localState) < prob){
                    selectedIndex = currIndex;
                }
                lastIndex = currIndex;
                counter++;
            }
            if(selectedIndex < 0){
                selectedIndex=lastIndex;
                if(lastIndex < 0){
                    selectedIndex = 0;
                }
            }
        }
        pPtsIndices[currentSampleIndex] = selectedIndex;
    }
}

////////////////////////////////////////////////////////////////////////////////// CPU

void NeighborRandSelectCPU(
    const bool pUseFeatures,
    const bool pScaleInv, 
    const int pdf,
    const float radius,
    const int pNumSamples,
    const int pNumNeighbors,
    const int pNumFeatures,
    const float* pInPts,
    const float* pInFeatures,
    const float* pInSamples,
    const float* pInSampleFeatures,
    const int* pBatchIds,
    const int* pStartIndexs,
    const int* pPackedIndexs,
    const float* pAABBMin,
    const float* pAABBMax,
    int* pPtsIndices)
{

    //gpuErrchk(cudaMemset(pPtsIndices, 0, sizeof(int)*pNumSamples));
    
    dim3 gridDimension = computeBlockGrid(pNumSamples, SAMLPES_BLOCK_SIZE);
    
    compute_rand_pts<<<gridDimension, SAMLPES_BLOCK_SIZE>>>(
        pUseFeatures, time(NULL), pScaleInv, pdf, radius, pNumSamples, pNumNeighbors, pNumFeatures, pInPts, 
        pInFeatures, pInSamples, pInSampleFeatures, pBatchIds, pStartIndexs, pPackedIndexs, pAABBMin, 
        pAABBMax, pPtsIndices);

    gpuErrchk(cudaPeekAtLastError());

}