#include <THC/THC.h>
#include <math.h>
#include <torch/extension.h>
#include "roi_align_kernel.h"


int roi_align_forward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                           torch::Tensor features, torch::Tensor rois, torch::Tensor output)
{
    // Grab the input tensor
    //float * data_flat = THCudaTensor_data(state, features);
    //float * rois_flat = THCudaTensor_data(state, rois);

    //float * output_flat = THCudaTensor_data(state, output);

    auto data_flat = features.data<float>();
    auto rois_flat = rois.data<float>();
    auto output_flat = output.data<float>();

    // Number of ROIs
    //int num_rois = THCudaTensor_size(state, rois, 0);
    //int size_rois = THCudaTensor_size(state, rois, 1);
    auto rois_sz = rois.sizes();
    int num_rois = rois_sz[0];
    int size_rois = rois_sz[1];
    if (size_rois != 5)
    {
        return 0;
    }

    // data height
    //int data_height = THCudaTensor_size(state, features, 2);
    // data width
    //int data_width = THCudaTensor_size(state, features, 3);
    // Number of channels
    //int num_channels = THCudaTensor_size(state, features, 1);
    auto feat_sz = features.sizes();
    int data_height = feat_sz[2];
    int data_width = feat_sz[3];
    int num_channels = feat_sz[1];


    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    ROIAlignForwardLaucher(
        data_flat, spatial_scale, num_rois, data_height,
        data_width, num_channels, aligned_height,
        aligned_width, rois_flat,
        output_flat, stream);

    return 1;
}

int roi_align_backward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                            torch::Tensor top_grad, torch::Tensor rois, torch::Tensor bottom_grad)
{
    // Grab the input tensor
    //float * top_grad_flat = THCudaTensor_data(state, top_grad);
    //float * rois_flat = THCudaTensor_data(state, rois);

    //float * bottom_grad_flat = THCudaTensor_data(state, bottom_grad);
    auto top_grad_flat = top_grad.data<float>();
    auto rois_flat = rois.data<float>();
    auto bottom_grad_flat = bottom_grad.data<float>();

    // Number of ROIs
    //int num_rois = THCudaTensor_size(state, rois, 0);
    //int size_rois = THCudaTensor_size(state, rois, 1);
    auto rois_sz = rois.sizes();
    int num_rois = rois_sz[0];
    int size_rois = rois_sz[1];

    if (size_rois != 5)
    {
        return 0;
    }

    // batch size
    //int batch_size = THCudaTensor_size(state, bottom_grad, 0);
    // data height
    //int data_height = THCudaTensor_size(state, bottom_grad, 2);
    // data width
    //int data_width = THCudaTensor_size(state, bottom_grad, 3);
    // Number of channels
    //int num_channels = THCudaTensor_size(state, bottom_grad, 1);
    auto grad_sz = bottom_grad.sizes();
    int batch_size = grad_sz[0];
    int data_height = grad_sz[2];
    int data_width = grad_sz[3];
    int num_channels = grad_sz[1];

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ROIAlignBackwardLaucher(
        top_grad_flat, spatial_scale, batch_size, num_rois, data_height,
        data_width, num_channels, aligned_height,
        aligned_width, rois_flat,
        bottom_grad_flat, stream);

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &roi_align_forward_cuda, "roi_align forward");
  m.def("backward", &roi_align_backward_cuda, "roi_align backward");
}
