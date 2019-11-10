/////////////////////////////////////////////////////////////////////////////
/// \file point_to_mesh_distance.cc
///
/// \brief C++ operation definition to ...
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

#include "cuda_kernel_utils.h"

using namespace tensorflow;

REGISTER_OP("PointToMeshDistance")
    .Input("points: float32")
    .Input("mesh_verts: float32")
    .Input("mesh_faces: int32")
    .Input("vox_face_indexs: int32")
    .Input("vox_voxel_indexs: int32")
    .Input("aabb_min: float32")
    .Input("cell_size: float32")
    .Output("distances: float32")
    .Output("closest_points: float32")
    .Output("triangle_index: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle outputDims = c->MakeShape({c->Dim(c->input(0), 0), 1});
        c->set_output(0, outputDims);
        c->set_output(1, c->input(0));
        c->set_output(2, outputDims);
        return Status::OK();
    });

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
    int* pTrianIndexs);

class PointToMeshDistanceOp : public OpKernel {
    public:
        explicit PointToMeshDistanceOp(OpKernelConstruction* context) : OpKernel(context) 
        { 
        }

        void Compute(OpKernelContext* context) override {
            
            const Tensor& inPointsTensor = context->input(0);
            OP_REQUIRES(context, inPointsTensor.dims() == 2, errors::InvalidArgument
                ("PointToMeshDistanceOp expects points with the following dimensions (batchSize, pointComponents)"));
            OP_REQUIRES(context, inPointsTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("PointToMeshDistanceOp expects points with three components"));
            int numPoints = inPointsTensor.shape().dim_size(0);
            auto inPointsFlat = inPointsTensor.flat<float>();
            const float* inPointsPtr = &(inPointsFlat(0));

            const Tensor& inMeshVertsTensor = context->input(1);
            OP_REQUIRES(context, inMeshVertsTensor.dims() == 2, errors::InvalidArgument
                ("PointToMeshDistanceOp expects mesh vertexs with the following dimensions (batchSize, pointComponents)"));
            OP_REQUIRES(context, inMeshVertsTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("PointToMeshDistanceOp expects mesh vertexs with three components"));
            int numMeshVertexs = inMeshVertsTensor.shape().dim_size(0);
            auto inMeshVertsFlat = inMeshVertsTensor.flat<float>();
            const float* inMeshVertsPtr = &(inMeshVertsFlat(0));

            const Tensor& inMeshFacesTensor=context->input(2);
            OP_REQUIRES(context, inMeshFacesTensor.dims() == 2 &&
                inMeshFacesTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("PointToMeshDistanceOp mesh faces with 3 vertices."));
            int numFaces = inMeshFacesTensor.shape().dim_size(0);
            auto inMeshFacesFlat = inMeshFacesTensor.flat<int>();
            const int* inMeshFacesPtr = &(inMeshFacesFlat(0));

            const Tensor& inVoxFacesIndexsTensor=context->input(3);
            OP_REQUIRES(context, inVoxFacesIndexsTensor.dims() == 1, errors::InvalidArgument
                ("PointToMeshDistanceOp faces index for each voxel with one dimension."));
            int numFacesVoxel = inVoxFacesIndexsTensor.shape().dim_size(0);
            auto inVoxFacesIndexsFlat = inVoxFacesIndexsTensor.flat<int>();
            const int* inVoxFacesIndexsPtr = &(inVoxFacesIndexsFlat(0));

            const Tensor& inVoxVoxelIndexsTensor=context->input(4);
            OP_REQUIRES(context, inVoxVoxelIndexsTensor.dims() == 4 &&
                inVoxVoxelIndexsTensor.shape().dim_size(3) == 2, errors::InvalidArgument
                ("PointToMeshDistanceOp voxel indexs for each voxel with four dimension."));
            int numVoxels[3] = {(int)inVoxVoxelIndexsTensor.shape().dim_size(0),
                                (int)inVoxVoxelIndexsTensor.shape().dim_size(1),
                                (int)inVoxVoxelIndexsTensor.shape().dim_size(2)}; 
            auto inVoxVoxelIndexsFlat = inVoxVoxelIndexsTensor.flat<int>();
            const int* inVoxVoxelIndexsPtr = &(inVoxVoxelIndexsFlat(0));

            const Tensor& inAABBMinTensor = context->input(5);
            OP_REQUIRES(context, inAABBMinTensor.dims() == 1 && 
                inAABBMinTensor.shape().dim_size(0) == 3, errors::InvalidArgument
                ("PointToMeshDistanceOp expects a minimum point of the bounding box with 3 components"));
            auto inAABBMinFlat = inAABBMinTensor.flat<float>();
            const float* inAABBMinPtr = &(inAABBMinFlat(0));

            const Tensor& inCellSizeTensor = context->input(6);
            OP_REQUIRES(context, inCellSizeTensor.dims() == 1 &&
                inCellSizeTensor.shape().dim_size(0) == 1, errors::InvalidArgument
                ("PointToMeshDistanceOp expects a cell size with 1 component"));
            auto inCellSizeFlat = inCellSizeTensor.flat<float>();
            const float* inCellSizePtr = &(inCellSizeFlat(0));

            Tensor* distances = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0, TensorShape{numPoints, 1}, &distances));
            auto distancesFlat = distances->flat<float>();
            float* distancesPtr = &(distancesFlat(0));
            Tensor* closestPts = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(1, TensorShape{numPoints, 3}, &closestPts));
            auto closestPtsFlat = closestPts->flat<float>();
            float* closestPtsPtr = &(closestPtsFlat(0));
            Tensor* trianIndexs = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(2, TensorShape{numPoints, 1}, &trianIndexs));
            auto trianIndexsFlat = trianIndexs->flat<int>();
            int* trianIndexsPtr = &(trianIndexsFlat(0));

            computeDistances(numPoints, numMeshVertexs, numFaces, numFacesVoxel, numVoxels, inPointsPtr, 
                inMeshVertsPtr, inMeshFacesPtr, inVoxFacesIndexsPtr, inVoxVoxelIndexsPtr, inAABBMinPtr,
                inCellSizePtr, distancesPtr, closestPtsPtr, trianIndexsPtr);
        }
};

REGISTER_KERNEL_BUILDER(Name("PointToMeshDistance").Device(DEVICE_GPU), PointToMeshDistanceOp);