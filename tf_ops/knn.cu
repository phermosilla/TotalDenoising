/////////////////////////////////////////////////////////////////////////////
/// \file compute_pdf.cu
///
/// \brief Cuda implementation of the operation to approximate the  
///        probability distribution function at each sample in the different  
///        receptive fields.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <iostream>
#include <fstream>

#include "cuda_kernel_utils.h"

#define SAMLPES_BLOCK_SIZE 64

////////////////////////////////////////////////////////////////////////////////// GPU


__global__ void compute_knn(
    const int knn,
    const bool farthest,
    const int pNumSamples,
    const int pNumNeighbors,
    const float* __restrict__ pInPts,
    const float* __restrict__ pInSamples,
    const int* __restrict__ pStartIndexs,
    const int* __restrict__ pPackedIndexs,
    int* __restrict__ pKNNIndices)
{
    extern __shared__ char auxiliarKNNList[];

    int currentSampleIndex = threadIdx.x + blockDim.x*(blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y);
    if(currentSampleIndex < pNumSamples){
        int initIter = pStartIndexs[currentSampleIndex];
        int endIter = (currentSampleIndex < pNumSamples-1)?pStartIndexs[currentSampleIndex+1]:pNumNeighbors;

        float currPointCoords[3] = {
            pInSamples[currentSampleIndex*3], 
            pInSamples[currentSampleIndex*3+1], 
            pInSamples[currentSampleIndex*3+2]};

        int* knnIndices = (int*)&auxiliarKNNList[threadIdx.x*sizeof(int)*knn*2];
        float* knnDistances = (float*)&auxiliarKNNList[threadIdx.x*sizeof(int)*knn*2+sizeof(int)*knn];

        for(int i = 0; i < knn; ++i)
        {
            knnIndices[i] = -1;
            knnDistances[i] = 0.0;
        }

        int iter = initIter;
        while(iter < endIter)
        {
            int iterPoint = pPackedIndexs[iter*2];
            float iterPointCoords[3] = {
                pInPts[iterPoint*3], 
                pInPts[iterPoint*3+1], 
                pInPts[iterPoint*3+2]};
            float diff [3] = {
                (iterPointCoords[0] - currPointCoords[0]), 
                (iterPointCoords[1] - currPointCoords[1]), 
                (iterPointCoords[2] - currPointCoords[2])};
            float distance = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2];
            
            int auxIndex = 0;
            float auxDistance = 0.0;
            int auxIndex2 = 0;
            float auxDistance2 = 0.0;
            bool movePoints = false;
            bool update = false;
            for(int i = 0; i < knn; ++i)
            {
                if(movePoints){
                    auxIndex2 = knnIndices[i];
                    auxDistance2 = knnDistances[i];
                    knnIndices[i] = auxIndex;
                    knnDistances[i] = auxDistance;
                    auxIndex = auxIndex2;
                    auxDistance = auxDistance2;
                }else{
                    update = false;
                    if(farthest){
                        update = knnDistances[i] < distance;
                    }else{
                        update = knnDistances[i] > distance;
                    }
                    if(knnIndices[i] < 0 || update){
                        auxIndex = knnIndices[i];
                        auxDistance = knnDistances[i];
                        movePoints = true;
                        knnIndices[i] = iterPoint;
                        knnDistances[i] = distance;
                    }
                }
            }
            iter++;
        }

        for(int i = 0; i < knn; ++i)
        {
            pKNNIndices[currentSampleIndex*knn + i] = knnIndices[i];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////// CPU

void computeKNNCPU(
    const int knn,
    const int pNumSamples,
    const int pNumNeighbors,
    const float* pInPts,
    const float* pInSamples,
    const int* pStartIndexs,
    const int* pPackedIndexs,
    int* pKNNIndices)
{
    int absKNN = abs(knn);
    bool farthest = knn < 0;
    gpuErrchk(cudaMemset(pKNNIndices, 0, sizeof(int)*pNumSamples*absKNN));
    
    dim3 gridDimension = computeBlockGrid(pNumSamples, SAMLPES_BLOCK_SIZE);

    compute_knn<<<gridDimension, SAMLPES_BLOCK_SIZE, absKNN*sizeof(int)*2*SAMLPES_BLOCK_SIZE>>>(
        absKNN, farthest, pNumSamples, pNumNeighbors, pInPts, pInSamples,
        pStartIndexs, pPackedIndexs, pKNNIndices);

    gpuErrchk(cudaPeekAtLastError());

}