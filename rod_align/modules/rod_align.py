from torch.nn.modules.module import Module
from torch.nn.functional import avg_pool2d, max_pool2d
from ..functions.rod_align import RoDAlignFunction


class RoDAlign(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoDAlign, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoDAlignFunction.apply(features,
                                      rois,
                                      self.aligned_height, 
                                      self.aligned_width,
                                      self.spatial_scale)

class RoDAlignAvg(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoDAlignAvg, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x =  RoDAlignFunction.apply(features,
                                    rois,
                                    self.aligned_height+1, 
                                    self.aligned_width+1,
                                    self.spatial_scale)
        return avg_pool2d(x, kernel_size=2, stride=1)

class RoDAlignMax(Module):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoDAlignMax, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x =  RoDAlignFunction.apply(features,
                                    rois,
                                    self.aligned_height+1, 
                                    self.aligned_width+1,
                                    self.spatial_scale)
        return max_pool2d(x, kernel_size=2, stride=1)
