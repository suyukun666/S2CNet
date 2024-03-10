#ifndef ROD_ALIGN_H
#define ROD_ALIGN_H

#include <torch/extension.h>

int rod_align_forward(int aligned_height, int aligned_width, float spatial_scale,
                      torch::Tensor features, torch::Tensor rois, torch::Tensor output);

int rod_align_backward(int aligned_height, int aligned_width, float spatial_scale,
                       torch::Tensor top_grad, torch::Tensor rois, torch::Tensor bottom_grad);

#endif
