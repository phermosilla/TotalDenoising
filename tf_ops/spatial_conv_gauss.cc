/////////////////////////////////////////////////////////////////////////////
/// \file spatial_conv_mlp1.cc
///
/// \brief C++ operations definition to perform a spatial convolution on a
///        batch of point clouds.
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

REGISTER_OP("SpatialConvGauss")
    .Attr("batch_size: int")
    .Attr("radius: float")
    .Attr("scale_inv: bool")
    .Input("points: float32")
    .Input("features: float32")
    .Input("batch_ids: int32")
    .Input("pdfs: float32")
    .Input("sample_pts: float32")
    .Input("start_neighs_indexs: int32")
    .Input("neighs_indexs: int32")
    .Input("aabb_min: float32")
    .Input("aabb_max: float32")
    .Output("out_features: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));
        return Status::OK();
    });

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
    float* pTmpBuff1,
    float* pTmpBuff2);


class SpatialConvGaussOp : public OpKernel {
    public:
        explicit SpatialConvGaussOp(OpKernelConstruction* context) : OpKernel(context) 
        {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0.0, errors::InvalidArgument("SpatialConvOp expects a positive radius"));  

            OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batchSize_));
            OP_REQUIRES(context, batchSize_ > 0, errors::InvalidArgument("SpatialConvOp expects a positive batch size"));

            OP_REQUIRES_OK(context, context->GetAttr("scale_inv", &scaleInv_));
        }

        void Compute(OpKernelContext* context) override {

            //Process input points.
            const Tensor& inPointsTensor = context->input(0);
            OP_REQUIRES(context, inPointsTensor.dims() == 2, errors::InvalidArgument
                ("SpatialConvOp expects points with the following dimensions (batchSize, pointComponents)"));
            OP_REQUIRES(context, inPointsTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SpatialConvOp expects points with three components"));
            int numPoints = inPointsTensor.shape().dim_size(0);
            auto inPointsFlat = inPointsTensor.flat<float>();
            const float* inPointsPtr = &(inPointsFlat(0));

            const Tensor& inFeaturesTensor=context->input(1);
            OP_REQUIRES(context, inFeaturesTensor.dims() == 2 &&
                inFeaturesTensor.shape().dim_size(0) == inPointsTensor.shape().dim_size(0), errors::InvalidArgument
                ("SpatialConvOp expects as feature inputs the following dimensions (numPoints)"));
            int numInFeatures = inFeaturesTensor.shape().dim_size(1);
            auto inFeaturesFlat = inFeaturesTensor.flat<float>();
            const float* inFeaturesPtr = &(inFeaturesFlat(0));

            const Tensor& batchIdsTensor = context->input(2); 
            OP_REQUIRES(context, batchIdsTensor.dims() == 2 && 
                batchIdsTensor.shape().dim_size(1) == 1 && 
                batchIdsTensor.shape().dim_size(0) == numPoints, errors::InvalidArgument
                ("SpatialConvOp expects correct btch ids"));
            auto batchIdsFlat = batchIdsTensor.flat<int>();
            const int* batchIdsPtr = &(batchIdsFlat(0));

            const Tensor& inPDFsTensor=context->input(3);
            OP_REQUIRES(context, inPDFsTensor.dims() == 2 &&
                inPDFsTensor.shape().dim_size(1) == 1, errors::InvalidArgument
                ("SpatialConvOp expects as feature inputs the following dimensions (numPoints)"));
            int numNeighs = inPDFsTensor.shape().dim_size(0);
            auto inPDFsTensorFlat = inPDFsTensor.flat<float>();
            const float* inPDFsTensorPtr = &(inPDFsTensorFlat(0));

            const Tensor& inSamplesTensor = context->input(4);
            OP_REQUIRES(context, inSamplesTensor.dims() == 2, errors::InvalidArgument
                ("SpatialConvOp expects points with the following dimensions (batchSize, pointComponents)"));
            OP_REQUIRES(context, inSamplesTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SpatialConvOp expects points with three components"));
            int numSamples = inSamplesTensor.shape().dim_size(0);
            auto inSamplesFlat = inSamplesTensor.flat<float>();
            const float* inSamplesPtr = &(inSamplesFlat(0));

            //Process start indexs.
            const Tensor& startIndexTensor = context->input(5); 
            OP_REQUIRES(context, startIndexTensor.dims() == 2 && 
                startIndexTensor.shape().dim_size(1) == 1 && 
                startIndexTensor.shape().dim_size(0) == numSamples, errors::InvalidArgument
                ("SpatialConvOp expects a four dimension tensor for the cell indices"));
            auto startIndexTensorFlat = startIndexTensor.flat<int>();
            const int* startIndexTensorPtr = &(startIndexTensorFlat(0));

            //Process packed neighbors.
            const Tensor& packedNeighTensor = context->input(6); 
            OP_REQUIRES(context, packedNeighTensor.dims() == 2 && 
                packedNeighTensor.shape().dim_size(0) == numNeighs &&
                packedNeighTensor.shape().dim_size(1) == 2, errors::InvalidArgument
                ("SpatialConvOp expects a four dimension tensor for the cell indices"));
            auto packedNeighTensorFlat = packedNeighTensor.flat<int>();
            const int* packedNeighTensorPtr = &(packedNeighTensorFlat(0));

            //Process input bounding box.
            const Tensor& inAABBMinTensor = context->input(7);
            OP_REQUIRES(context, inAABBMinTensor.dims() == 2 
                && inAABBMinTensor.shape().dim_size(0) == batchSize_ && inAABBMinTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SpatialConvOp expects a minimum point of the bounding box with 3 components"));
            auto inAABBMinFlat = inAABBMinTensor.flat<float>();
            const float* inAABBMinPtr = &(inAABBMinFlat(0));

            const Tensor& inAABBMaxTensor = context->input(8);
            OP_REQUIRES(context, inAABBMaxTensor.dims() == 2  
                && inAABBMaxTensor.shape().dim_size(0) == batchSize_ && inAABBMaxTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SpatialConvOp expects a maximum point of the bounding box with 3 components"));
            auto inAABBMaxFlat = inAABBMaxTensor.flat<float>();
            const float* inAABBMaxPtr = &(inAABBMaxFlat(0));

            //Create the temporal buffer.
            Tensor tmpBuff;
            OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{numNeighs}, &tmpBuff));
            auto tmpBuffFlat = tmpBuff.flat<float>();
            float* tmpBuffPtr = &(tmpBuffFlat(0));
            Tensor tmpBuff2;
            OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{numSamples}, &tmpBuff2));
            auto tmpBuff2Flat = tmpBuff2.flat<float>();
            float* tmpBuff2Ptr = &(tmpBuff2Flat(0));

            //Create the output tensors.
            Tensor* outConvFeatures = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0, TensorShape{numSamples, numInFeatures}, &outConvFeatures));
            auto outConvFeaturesFlat = outConvFeatures->flat<float>();
            float* outConvFeaturesPtr = &(outConvFeaturesFlat(0));

            spatialConvGaussCPU(scaleInv_, numNeighs, numInFeatures, numSamples, radius_, inPointsPtr, 
                batchIdsPtr,  inFeaturesPtr, inPDFsTensorPtr, inSamplesPtr, startIndexTensorPtr, packedNeighTensorPtr, 
                inAABBMinPtr, inAABBMaxPtr, outConvFeaturesPtr, tmpBuffPtr, tmpBuff2Ptr);
        }

    private:

        float   radius_;
        int     batchSize_;
        bool    scaleInv_;
};

REGISTER_KERNEL_BUILDER(Name("SpatialConvGauss").Device(DEVICE_GPU), SpatialConvGaussOp);