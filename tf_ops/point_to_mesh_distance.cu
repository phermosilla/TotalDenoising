/////////////////////////////////////////////////////////////////////////////
/// \file point_to_mesh_distance.cu
///
/// \brief Cuda implementation of the operation ...
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
#include <vector_types.h>
#include <vector_functions.h>

#define POINTS_BLOCK_SIZE 64

////////////////////////////////////////////////////////////////////////////////// GPU

__constant__ int cellOffsetsPt2Mesh[27][3];

#define ASSIGN_FLOAT4(pDest, pSource)\
    pDest.x = pSource.x; \
    pDest.y = pSource.y; \
    pDest.z = pSource.z; \
    pDest.w = pSource.w; 

#define ADD_FLOAT4(pDest, pSource1, pSource2)\
    pDest.x = pSource1.x + pSource2.x; \
    pDest.y = pSource1.y + pSource2.y; \
    pDest.z = pSource1.z + pSource2.z; \
    pDest.w = pSource1.w + pSource2.w; 

#define SUB_FLOAT4(pDest, pSource1, pSource2)\
    pDest.x = pSource1.x - pSource2.x; \
    pDest.y = pSource1.y - pSource2.y; \
    pDest.z = pSource1.z - pSource2.z; \
    pDest.w = pSource1.w - pSource2.w; 

#define ADD_MUL_FLOAT4(pDest, pSource1, pSource2, pScalar)\
    pDest.x = pSource1.x + pSource2.x*pScalar; \
    pDest.y = pSource1.y + pSource2.y*pScalar; \
    pDest.z = pSource1.z + pSource2.z*pScalar; \
    pDest.w = pSource1.w + pSource2.w*pScalar; 

#define DOT_FLOAT4(pSource1, pSource2)\
    pSource1.x * pSource2.x + \
    pSource1.y * pSource2.y + \
    pSource1.z * pSource2.z + \
    pSource1.w * pSource2.w

#define ZERO_EPSILON 0.0000001

__device__ float4 trianglePointDistance(
    const float4& pVert1,
    const float4& pVert2,
    const float4& pVert3,
    const float4& pPoint)
{
    float4 result, diffResult, ab, ac, bc, ap, bp, cp;

    SUB_FLOAT4(ab, pVert2, pVert1);
    SUB_FLOAT4(ac, pVert3, pVert1);
    SUB_FLOAT4(bc, pVert3, pVert2);
    SUB_FLOAT4(ap, pPoint, pVert1);
    float d1 = DOT_FLOAT4(ab, ap);
    float d2 = DOT_FLOAT4(ac, ap);
    if(d1 <= 0.0f && d2 <= 0.0f){
        ASSIGN_FLOAT4(result, pVert1);
        SUB_FLOAT4(diffResult, result, pPoint);
        result.w = DOT_FLOAT4(diffResult, diffResult);
        return result;
    } 

    SUB_FLOAT4(bp, pPoint, pVert2);
    float d3 = DOT_FLOAT4(ab, bp);
    float d4 = DOT_FLOAT4(ac, bp);
    if(d3 >= 0.0f && d4 <= d3){
        ASSIGN_FLOAT4(result, pVert2);
        SUB_FLOAT4(diffResult, result, pPoint);
        result.w = DOT_FLOAT4(diffResult, diffResult);
        return result;
    } 

    SUB_FLOAT4(cp, pPoint, pVert3);
    float d5 = DOT_FLOAT4(ab, cp);
    float d6 = DOT_FLOAT4(ac, cp);
    if(d6 >= 0.0f && d5 <= d6){
        ASSIGN_FLOAT4(result, pVert3);
        SUB_FLOAT4(diffResult, result, pPoint);
        result.w = DOT_FLOAT4(diffResult, diffResult);
        return result;
    } 

    float vc = d1*d4 - d3*d2;
    if(vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f){
        float v1 = d1 / (d1-d3);
        ADD_MUL_FLOAT4(result, pVert1, ab, v1);
        SUB_FLOAT4(diffResult, result, pPoint);
        result.w = DOT_FLOAT4(diffResult, diffResult);
        return result;
    }

    float vb = d5*d2 - d1*d6;
    if(vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f){
        float w1 = d2 / (d2-d6);
        ADD_MUL_FLOAT4(result, pVert1, ac, w1);
        SUB_FLOAT4(diffResult, result, pPoint);
        result.w = DOT_FLOAT4(diffResult, diffResult);
        return result;
    }

    float va = d3*d6 - d5*d4;
    if(va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f){
        float w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        ADD_MUL_FLOAT4(result, pVert2, bc, w2);
        SUB_FLOAT4(diffResult, result, pPoint);
        result.w = DOT_FLOAT4(diffResult, diffResult);
        return result;
    }

    float denom = 1.0f/(va + vb + vc);
    float v2 = vb * denom;
    float w3 = vc * denom;
    ADD_MUL_FLOAT4(result, pVert1, ab, v2);
    ADD_MUL_FLOAT4(result, result, ac, w3);
    SUB_FLOAT4(diffResult, result, pPoint);
    result.w = DOT_FLOAT4(diffResult, diffResult);
    return result;
}

__global__ void compute_distances(
    const int pNumPoints,
    const int pNumVoxelsX,
    const int pNumVoxelsY,
    const int pNumVoxelsZ,
    const float* __restrict__ pPoints,
    const float* __restrict__ pVertexs,
    const int* __restrict__ pFaces,
    const int* __restrict__ pVoxeFaceIndexs,
    const int* __restrict__ pVoxelIndexs,
    const float* __restrict__ pAABBMin,
    const float* __restrict__ pCellSize,
    float* __restrict__ pOutDistances,
    float* __restrict__ pOutClosestPoints,
    int* __restrict__ pOutTrianIndexs) 
{
    int ptIndex = threadIdx.x + blockDim.x*(blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y);
    if(ptIndex < pNumPoints){
        float4 ptPos = make_float4(pPoints[ptIndex*3], pPoints[ptIndex*3+1], pPoints[ptIndex*3+2], 0.0f);
        int xCell = max(min((int)floor((ptPos.x - pAABBMin[0])/pCellSize[0]), pNumVoxelsX -1), 0);
        int yCell = max(min((int)floor((ptPos.y - pAABBMin[1])/pCellSize[0]), pNumVoxelsY -1), 0);
        int zCell = max(min((int)floor((ptPos.z - pAABBMin[2])/pCellSize[0]), pNumVoxelsZ -1), 0);
        float4 closestPt = make_float4(0.0f, 0.0f, 0.0f, -1.0f);
        int trianIndex = -1;
        for(int i = 0; i < 27; ++i)
        {
            int currCellIndex[3] = {xCell+cellOffsetsPt2Mesh[i][0], 
                                    yCell+cellOffsetsPt2Mesh[i][1], 
                                    zCell+cellOffsetsPt2Mesh[i][2]};
            if(currCellIndex[0] >= 0 && currCellIndex[0] < pNumVoxelsX &&
                currCellIndex[1] >= 0 && currCellIndex[1] < pNumVoxelsY &&
                currCellIndex[2] >= 0 && currCellIndex[2] < pNumVoxelsZ)
            {
                int linearCellIndex =   currCellIndex[0]*pNumVoxelsY*pNumVoxelsZ + 
                                        currCellIndex[1]*pNumVoxelsZ + currCellIndex[2];
                int startFaceIndex = pVoxelIndexs[linearCellIndex*2];
                int endFaceIndex = pVoxelIndexs[linearCellIndex*2+1];
                if(startFaceIndex > 0){
                    for(int iter = startFaceIndex; iter < endFaceIndex; ++iter)
                    {
                        int currFaceIndex = pVoxeFaceIndexs[iter];
                        int currFace[3] = { pFaces[currFaceIndex*3],
                                            pFaces[currFaceIndex*3+1],
                                            pFaces[currFaceIndex*3+2]};
                        float4 vertex1 = make_float4(pVertexs[currFace[0]*3],
                                                    pVertexs[currFace[0]*3+1],
                                                    pVertexs[currFace[0]*3+2],
                                                    0.0f);
                        float4 vertex2 = make_float4(pVertexs[currFace[1]*3],
                                                    pVertexs[currFace[1]*3+1],
                                                    pVertexs[currFace[1]*3+2],
                                                    0.0f);
                        float4 vertex3 = make_float4(pVertexs[currFace[2]*3],
                                                    pVertexs[currFace[2]*3+1],
                                                    pVertexs[currFace[2]*3+2],
                                                    0.0f);
                        float4 result = trianglePointDistance(vertex1, vertex2, vertex3, ptPos);
                        if(result.w >= 0.0f && (result.w < closestPt.w || closestPt.w < 0.0f)){
                            ASSIGN_FLOAT4(closestPt, result);
                            trianIndex = currFaceIndex;
                        }
                    }
                }
            }
        }
        if(closestPt.w < 0.0)
            closestPt.w = 1000.0f;
        pOutDistances[ptIndex] = sqrt(closestPt.w);
        pOutClosestPoints[ptIndex*3] = closestPt.x;
        pOutClosestPoints[ptIndex*3+1] = closestPt.y;
        pOutClosestPoints[ptIndex*3+2] = closestPt.z;
        pOutTrianIndexs[ptIndex] = trianIndex;
    }
}

////////////////////////////////////////////////////////////////////////////////// CPU

void computeDistances(
    const int pNumPts,
    const int pNumVertexs,
    const int pNumFaces,
    const int pNumVoxFaceIndexs,
    const int pNumVoxels[3],
    const float* pInPoints,
    const float* pInVertexs,
    const int* pInFaces,
    const int* pVoxFaceIndexs,
    const int* pVoxVoxelIndexs,
    const float* pAABBMin,
    const float* pCellSize,
    float* pOutDistances,
    float* pOutClosestPoints,
    int* pTrianIndexs)
{    
    int cellOffsetsCPU[27][3] = {
        {1, 1, 1},{0, 1, 1},{-1, 1, 1},
        {1, 0, 1},{0, 0, 1},{-1, 0, 1},
        {1, -1, 1},{0, -1, 1},{-1, -1, 1},
        {1, 1, 0},{0, 1, 0},{-1, 1, 0},
        {1, 0, 0},{0, 0, 0},{-1, 0, 0},
        {1, -1, 0},{0, -1, 0},{-1, -1, 0},
        {1, 1, -1},{0, 1, -1},{-1, 1, -1},
        {1, 0, -1},{0, 0, -1},{-1, 0, -1},
        {1, -1, -1},{0, -1, -1},{-1, -1, -1}};
    cudaMemcpyToSymbol(cellOffsetsPt2Mesh, cellOffsetsCPU, 27*3*sizeof(int));

    dim3 gridDimension = computeBlockGrid(pNumPts, POINTS_BLOCK_SIZE);
    compute_distances<<<gridDimension, POINTS_BLOCK_SIZE>>>(pNumPts, 
        pNumVoxels[0], pNumVoxels[1], pNumVoxels[2], pInPoints, pInVertexs, pInFaces, 
        pVoxFaceIndexs, pVoxVoxelIndexs, pAABBMin, pCellSize, pOutDistances, 
        pOutClosestPoints, pTrianIndexs);

    gpuErrchk(cudaPeekAtLastError());
   
}