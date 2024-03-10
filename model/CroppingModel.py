import torch
import torch.nn as nn
import torchvision.models as models
from roi_align.modules.roi_align import RoIAlignAvg, RoIAlign
from rod_align.modules.rod_align import RoDAlignAvg, RoDAlign
import torch.nn.init as init
from model.mobilenetv2 import MobileNetV2


class efficientnet_base(nn.Module):
    def __init__(self, loadweights=True, downsample=4):
        super(efficientnet_base, self).__init__()

        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V2)
        self.feat_ext = model.features
        self.feature3 = nn.Sequential(model.features[:4])
        self.feature4 = nn.Sequential(model.features[4:6])
        self.feature5 = nn.Sequential(model.features[6:])

    def forward(self, x):
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5


class mobilenetv2_base(nn.Module):

    def __init__(self, loadweights=True, downsample=4, model_path='./pretrained_model/mobilenetv2_1.0-0c6065bc.pth'):
        super(mobilenetv2_base, self).__init__()

        model = MobileNetV2(width_mult=1.0)

        if loadweights:
            model.load_state_dict(torch.load(model_path))

        self.feature3 = nn.Sequential(model.features[:7])
        self.feature4 = nn.Sequential(model.features[7:14])
        self.feature5 = nn.Sequential(model.features[14:])

    def forward(self, x):
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5

