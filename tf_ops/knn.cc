/////////////////////////////////////////////////////////////////////////////
/// \file knn.cc
///
/// \brief C++ operation definition to compute the knn of a set of points.
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

REGISTER_OP("Knn")
    .Attr("knn: int")
    .Input("points: float32")
    .Input("samples: float32")
    .Input("start_indexs: int32")
    .Input("neigbors: int32")
    .Output("knn_indices: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        int knn;
        TF_RETURN_IF_ERROR(c->GetAttr("knn", &knn));
        shape_inference::ShapeHandle outputDims = c->MakeShape({c->Dim(c->input(1), 0), abs(knn)});
        c->set_output(0, outputDims);
        return Status::OK();
    });

void computeKNNCPU(
    const int knn,
    const int pNumSamples,
    const int pNumNeighbors,
    const float* pInPts,
    const float* pInSamples,
    const int* pStartIndexs,
    const int* pPackedIndexs,
    int* pKNNIndices);

class KnnOp : public OpKernel {
    public:
        explicit KnnOp(OpKernelConstruction* context) : OpKernel(context) 
        { 

            OP_REQUIRES_OK(context, context->GetAttr("knn", &knn_));
            OP_REQUIRES(context, knn_ != 0, errors::InvalidArgument("KNNOp expects a non zero number of points"));  
        }

        void Compute(OpKernelContext* context) override {
            //Process input points.
            const Tensor& inPointsTensor = context->input(0);
            OP_REQUIRES(context, inPointsTensor.dims() == 2, errors::InvalidArgument
                ("KNNOp expects points with the following dimensions (pointComponents, dimensions)"));
            OP_REQUIRES(context, inPointsTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("KNNOp expects points with three components"));
            int numPoints = inPointsTensor.shape().dim_size(0);
            auto inPointsFlat = inPointsTensor.flat<float>();
            const float* inPointsPtr = &(inPointsFlat(0));

            //Process input samples.
            const Tensor& inSamplesTensor = context->input(1);
            OP_REQUIRES(context, inSamplesTensor.dims() == 2, errors::InvalidArgument
                ("KNNOp expects points with the following dimensions (pointComponents, dimensions)"));
            OP_REQUIRES(context, inSamplesTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("KNNOp expects points with three components"));
            int numSamples = inSamplesTensor.shape().dim_size(0);
            auto inSamplesFlat = inSamplesTensor.flat<float>();
            const float* inSamplesPtr = &(inSamplesFlat(0));

            //Process start indexs.
            const Tensor& startIndexTensor = context->input(2); 
            OP_REQUIRES(context, startIndexTensor.dims() == 2 && 
                startIndexTensor.shape().dim_size(0) == numSamples &&
                startIndexTensor.shape().dim_size(1) == 1, errors::InvalidArgument
                ("KNNOp expects a correct start indices of the samples's neighbors"));
            auto startIndexTensorFlat = startIndexTensor.flat<int>();
            const int* startIndexTensorPtr = &(startIndexTensorFlat(0));

            //Process packed neighbors.
            const Tensor& packedNeighTensor = context->input(3); 
            OP_REQUIRES(context, packedNeighTensor.dims() == 2 && 
                packedNeighTensor.shape().dim_size(1) == 2, errors::InvalidArgument
                ("KNNOp expects a packed neighbors with 2 dimensions"));
            int numNeighs = packedNeighTensor.shape().dim_size(0);
            auto packedNeighTensorFlat = packedNeighTensor.flat<int>();
            const int* packedNeighTensorPtr = &(packedNeighTensorFlat(0));

            //Create the output tensors.
            Tensor* knnIndices = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0, TensorShape{numSamples, abs(knn_)}, &knnIndices));
            auto knnIndicesFlat = knnIndices->flat<int>();
            int* knnIndicesPtr = &(knnIndicesFlat(0));

            //Compute the knn
            computeKNNCPU(knn_, numSamples, numNeighs, inPointsPtr, inSamplesPtr, startIndexTensorPtr, 
                packedNeighTensorPtr, knnIndicesPtr);
        }

    private:

        int     knn_;
};

REGISTER_KERNEL_BUILDER(Name("Knn").Device(DEVICE_GPU), KnnOp);