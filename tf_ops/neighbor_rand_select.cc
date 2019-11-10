/////////////////////////////////////////////////////////////////////////////
/// \file neighbor_rand_select.cc
///
/// \brief C++ operation definition to select a random point from the 
///     receptive field.
///
/// \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
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

REGISTER_OP("NeighborRandSelect")
    .Attr("pdf: int")
    .Attr("radius: float")
    .Attr("batch_size: int")
    .Attr("scale_inv: bool")
    .Attr("use_features: bool")
    .Input("points: float32")
    .Input("features: float32")
    .Input("samples: float32")
    .Input("sample_features: float32")
    .Input("samples_batch_ids: int32")
    .Input("start_indexs: int32")
    .Input("neigbors: int32")
    .Input("aabb_min: float32")
    .Input("aabb_max: float32")
    .Output("pts_indices: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle outputDims = c->MakeShape({c->Dim(c->input(1), 0), 1});
        c->set_output(0, outputDims);
        return Status::OK();
    });

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
    int* pPtsIndices);

class NeighborRandSelectOp : public OpKernel {
    public:
        explicit NeighborRandSelectOp(OpKernelConstruction* context) : OpKernel(context) 
        { 

            OP_REQUIRES_OK(context, context->GetAttr("pdf", &pdf_));
            OP_REQUIRES(context, pdf_ >= 0 && pdf_ < 4, errors::InvalidArgument
                ("NeighborRandSelectOp expects a pdf type between [0,2]"));  
            
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0.0, errors::InvalidArgument
                ("NeighborRandSelectOp expects a radius_ greater than 0.0"));  

            OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batchSize_));
            OP_REQUIRES(context, batchSize_ > 0, errors::InvalidArgument("SpatialConvGradOp expects a positive batch size"));

            OP_REQUIRES_OK(context, context->GetAttr("scale_inv", &scaleInv_));

            OP_REQUIRES_OK(context, context->GetAttr("use_features", &useFeatures_));
                
        }

        void Compute(OpKernelContext* context) override {
            //Process input points.
            const Tensor& inPointsTensor = context->input(0);
            OP_REQUIRES(context, inPointsTensor.dims() == 2, errors::InvalidArgument
                ("NeighborRandSelectOp expects points with the following dimensions (pointComponents, dimensions)"));
            OP_REQUIRES(context, inPointsTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("NeighborRandSelectOp expects points with three components"));
            int numPoints = inPointsTensor.shape().dim_size(0);
            auto inPointsFlat = inPointsTensor.flat<float>();
            const float* inPointsPtr = &(inPointsFlat(0));

            //Process input features.
            const Tensor& inFeaturesTensor = context->input(1);
            OP_REQUIRES(context, inFeaturesTensor.dims() == 2, errors::InvalidArgument
                ("NeighborRandSelectOp expects features with the following dimensions (pointComponents, numFeatures)"));
            OP_REQUIRES(context, inFeaturesTensor.shape().dim_size(0) == numPoints, errors::InvalidArgument
                ("NeighborRandSelectOp expects features with the following dimensions (pointComponents, numFeatures)"));
            int numFeatures = inFeaturesTensor.shape().dim_size(1);
            auto inFeaturesFlat = inFeaturesTensor.flat<float>();
            const float* inFeaturesPtr = &(inFeaturesFlat(0));

            //Process input samples.
            const Tensor& inSamplesTensor = context->input(2);
            OP_REQUIRES(context, inSamplesTensor.dims() == 2, errors::InvalidArgument
                ("NeighborRandSelectOp expects points with the following dimensions (pointComponents, dimensions)"));
            OP_REQUIRES(context, inSamplesTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("NeighborRandSelectOp expects points with three components"));
            int numSamples = inSamplesTensor.shape().dim_size(0);
            auto inSamplesFlat = inSamplesTensor.flat<float>();
            const float* inSamplesPtr = &(inSamplesFlat(0));

            //Process input sample features.
            const Tensor& inSampleFeaturesTensor = context->input(3);
            OP_REQUIRES(context, inSampleFeaturesTensor.dims() == 2, errors::InvalidArgument
                ("NeighborRandSelectOp expects features with the following dimensions (pointComponents, numFeatures)"));
            OP_REQUIRES(context, inSampleFeaturesTensor.shape().dim_size(0) == numPoints &&
                inSampleFeaturesTensor.shape().dim_size(1) == numFeatures, errors::InvalidArgument
                ("NeighborRandSelectOp expects features with the following dimensions (pointComponents, numFeatures)"));
            auto inSampleFeaturesFlat = inSampleFeaturesTensor.flat<float>();
            const float* inSampleFeaturesPtr = &(inSampleFeaturesFlat(0));

            //Process input samples batchIds.
            const Tensor& inSamplesBatchIdsTensor = context->input(4);
            OP_REQUIRES(context, inSamplesBatchIdsTensor.dims() == 2, errors::InvalidArgument
                ("NeighborRandSelectOp expects points with the following dimensions (pointComponents, dimensions)"));
            OP_REQUIRES(context, inSamplesBatchIdsTensor.shape().dim_size(0) == numSamples, errors::InvalidArgument
                ("NeighborRandSelectOp expects points with three components"));
            OP_REQUIRES(context, inSamplesBatchIdsTensor.shape().dim_size(1) == 1, errors::InvalidArgument
                ("NeighborRandSelectOp expects points with three components"));
            auto inSamplesBatchIdsFlat = inSamplesBatchIdsTensor.flat<int>();
            const int* inSamplesBatchIdsPtr = &(inSamplesBatchIdsFlat(0));

            //Process start indexs.
            const Tensor& startIndexTensor = context->input(5); 
            OP_REQUIRES(context, startIndexTensor.dims() == 2 && 
                startIndexTensor.shape().dim_size(0) == numSamples &&
                startIndexTensor.shape().dim_size(1) == 1, errors::InvalidArgument
                ("NeighborRandSelectOp expects a correct start indices of the samples's neighbors"));
            auto startIndexTensorFlat = startIndexTensor.flat<int>();
            const int* startIndexTensorPtr = &(startIndexTensorFlat(0));

            //Process packed neighbors.
            const Tensor& packedNeighTensor = context->input(6); 
            OP_REQUIRES(context, packedNeighTensor.dims() == 2 && 
                packedNeighTensor.shape().dim_size(1) == 2, errors::InvalidArgument
                ("NeighborRandSelectOp expects a packed neighbors with 2 dimensions"));
            int numNeighs = packedNeighTensor.shape().dim_size(0);
            auto packedNeighTensorFlat = packedNeighTensor.flat<int>();
            const int* packedNeighTensorPtr = &(packedNeighTensorFlat(0));

            //Process input bounding box.
            const Tensor& inAABBMinTensor = context->input(7);
            OP_REQUIRES(context, inAABBMinTensor.dims() == 2 
                && inAABBMinTensor.shape().dim_size(0) == batchSize_ && inAABBMinTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("NeighborRandSelectOp expects a minimum point of the bounding box with 3 components"));
            auto inAABBMinFlat = inAABBMinTensor.flat<float>();
            const float* inAABBMinPtr = &(inAABBMinFlat(0));

            const Tensor& inAABBMaxTensor = context->input(8);
            OP_REQUIRES(context, inAABBMaxTensor.dims() == 2  
                && inAABBMaxTensor.shape().dim_size(0) == batchSize_ && inAABBMaxTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("NeighborRandSelectOp expects a maximum point of the bounding box with 3 components"));
            auto inAABBMaxFlat = inAABBMaxTensor.flat<float>();
            const float* inAABBMaxPtr = &(inAABBMaxFlat(0));

            //Create the output tensors.
            Tensor* ptsIndices = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0, TensorShape{numSamples, 1}, &ptsIndices));
            auto ptsIndicesFlat = ptsIndices->flat<int>();
            int* ptsIndicesPtr = &(ptsIndicesFlat(0));

            //Compute the random points
            NeighborRandSelectCPU(useFeatures_, 
                scaleInv_, pdf_, radius_, numSamples, numNeighs, numFeatures, inPointsPtr, inFeaturesPtr,
                inSamplesPtr, inSampleFeaturesPtr, inSamplesBatchIdsPtr, startIndexTensorPtr, 
                packedNeighTensorPtr, inAABBMinPtr, inAABBMaxPtr, ptsIndicesPtr);
        }

    private:

        int     pdf_;
        float   radius_;
        int     batchSize_;
        bool    scaleInv_;
        bool    useFeatures_;
};

REGISTER_KERNEL_BUILDER(Name("NeighborRandSelect").Device(DEVICE_GPU), NeighborRandSelectOp);