#ifndef ROI_ALIGN_CUDA_H
#define ROI_ALIGN_CUDA_H

#include <torch/extension.h>

int roi_align_forward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                           torch::Tensor features, torch::Tensor rois, torch::Tensor output);

int roi_align_backward_cuda(int aligned_height, int aligned_width, float spatial_scale,
                            torch::Tensor top_grad, torch::Tensor rois, torch::Tensor bottom_grad);

#endif
