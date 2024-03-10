import torch
from torch.autograd import Function
import roi_align_api

class RoIAlignFunction(Function):
    @staticmethod
    def forward(ctx, features, rois, aligned_height, aligned_width, spatial_scale):
        batch_size, num_channels, data_height, data_width = features.size()
        ctx.save_for_backward(rois, 
                              torch.IntTensor([int(batch_size),
                                               int(num_channels),
                                               int(data_height),
                                               int(data_width),
                                               int(aligned_height), 
                                               int(aligned_width)]),
                              torch.FloatTensor([float(spatial_scale)]))

        num_rois = rois.size(0)

        output = features.new(num_rois, 
                              num_channels, 
                              int(aligned_height), 
                              int(aligned_width)).zero_()
        
        roi_align_api.forward(int(aligned_height),
                              int(aligned_width),
                              float(spatial_scale), 
                              features,
                              rois,
                              output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        rois, core_size, scale = ctx.saved_tensors

        batch_size, num_channels, data_height, data_width, aligned_height, aligned_width = core_size
        spatial_scale = scale[0]

        grad_input = rois.new(batch_size,
                              num_channels, 
                              data_height,
                              data_width).zero_()

        roi_align_api.backward(int(aligned_height),
                               int(aligned_width),
                               float(spatial_scale), 
                               grad_output,
                               rois, 
                               grad_input)

        return grad_input, None, None, None, None
