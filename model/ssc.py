import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from model.CroppingModel import mobilenetv2_base, efficientnet_base
from rod_align.modules.rod_align import RoDAlignAvg
from roi_align.modules.roi_align import RoIAlignAvg
from model.attention import GraphAttention


def fc_layers(reddim, alignsize):
    # borrow from https://github.com/lld533/Grid-Anchor-based-Image-Cropping-Pytorch/blob/master/croppingModel.py
    conv1 = nn.Sequential(nn.Conv2d(reddim, 768, kernel_size=alignsize, padding=0),nn.BatchNorm2d(768),nn.ReLU(inplace=True))
    conv2 = nn.Sequential(nn.Conv2d(768, 128, kernel_size=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
    dropout = nn.Dropout(p=0.5)
    conv3 = nn.Conv2d(128, 1, kernel_size=1)
    layers = nn.Sequential(conv1, conv2, dropout, conv3)
    return layers

def xavier(param):
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

class GraphConstruction(nn.Module):
    def __init__(self, num_heads=4, num_feature_node=512, num_features_relation=256):
        super(GraphConstruction, self).__init__()
        self.heads = num_heads
        self.NFR = num_features_relation
        self.NFG = num_feature_node
        self.fc_rn_theta = nn.Linear(self.NFG, self.NFR)
        self.fc_rn_phi = nn.Linear(self.NFG, self.NFR)
        self.pos_ascension_theta = nn.Linear(2, self.NFR)
        self.pos_ascension_phi = nn.Linear(2, self.NFR)
        self.pos_to_head = nn.Linear(1, num_heads)
        self.app_to_head = nn.Linear(1, num_heads)

        
    def forward(self, graph_boxes_features, boxes_in_flat, height, width):
        """
        graph_boxes_features  [N,D+2,NFG] NFG=reddim*alignsize*alignsize
        """
        # graph modeling
        N, D, E = graph_boxes_features.shape
        
        # position encode
        graph_boxes_positions = torch.empty(N, D, 2).to(boxes_in_flat.device)
        graph_boxes_positions[:, :, 0] = (boxes_in_flat[:, :, 1] + boxes_in_flat[:, :, 3]) / 2 / width
        graph_boxes_positions[:, :, 1] = (boxes_in_flat[:, :, 2] + boxes_in_flat[:, :, 4]) / 2 / height
        graph_boxes_positions_theta = self.pos_ascension_theta(graph_boxes_positions) 
        graph_boxes_positions_phi = self.pos_ascension_phi(graph_boxes_positions) 

        # distance
        graph_boxes_distances = self.calc_pairwise_distance_3d(graph_boxes_positions_theta, graph_boxes_positions_phi) / np.sqrt(self.NFR) 

        # relation encode
        graph_boxes_features_theta = self.fc_rn_theta(graph_boxes_features) 
        graph_boxes_features_phi = self.fc_rn_phi(graph_boxes_features) 
        
        similarity_relation_graph = torch.matmul(graph_boxes_features_theta, graph_boxes_features_phi.transpose(1, 2))
        similarity_relation_graph = similarity_relation_graph / np.sqrt(self.NFR)

        graph_boxes_distances = self.pos_to_head(graph_boxes_distances.unsqueeze(-1)).permute(0, 3, 1, 2)
        similarity_relation_graph = self.app_to_head(similarity_relation_graph.unsqueeze(-1)).permute(0, 3, 1, 2)

        return graph_boxes_distances, similarity_relation_graph
        

    def calc_pairwise_distance_3d(self, X, Y):
        """
        computes pairwise distance between each element
        Args: 
            X: [B,N,D]
            Y: [B,M,D]
        Returns:
            dist: [B,N,M] matrix of euclidean distances
        """
        B=X.shape[0]
        rx=X.pow(2).sum(dim=2).reshape((B,-1,1))
        ry=Y.pow(2).sum(dim=2).reshape((B,-1,1))
        dist=rx-2.0*X.matmul(Y.transpose(1,2))+ry.transpose(1,2)
        return torch.sqrt(dist)


class SSC(nn.Module):

    def __init__(self, cfg):
        super(SSC, self).__init__()

        self.alignsize = cfg.align_size
        self.reddim = cfg.reduced_dim

        if cfg.base_model == 'mobilenetv2':
            self.Feat_ext = mobilenetv2_base(cfg.loadweight, cfg.downsample)
            self.DimRed = nn.Conv2d(448, self.reddim, kernel_size=1, padding=0)
        elif cfg.base_model == 'efficient':
            self.Feat_ext = efficientnet_base(cfg.loadweight, cfg.downsample)
            self.DimRed = nn.Conv2d(1432, self.reddim, kernel_size=1, padding=0)
        
        self.downsample2 = nn.UpsamplingBilinear2d(scale_factor=1.0/2.0)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.RoIAlign = RoIAlignAvg(self.alignsize, self.alignsize, 1.0/2**cfg.downsample)
        self.RoDAlign = RoDAlignAvg(self.alignsize, self.alignsize, 1.0/2**cfg.downsample)

        self.node_emb = nn.Sequential(
            nn.Linear(2*self.reddim*self.alignsize*self.alignsize, cfg.num_feature_node),
            nn.LayerNorm([cfg.num_feature_node]), nn.ReLU(inplace=True)
        )

        self.GraphConstructor = GraphConstruction(cfg.num_heads, cfg.num_feature_node, cfg.num_features_relation)
        self.GraphTransformer = GraphAttention(dim=cfg.num_feature_node, depth=cfg.num_depth, heads=cfg.num_heads, mlp_dim=cfg.mlp_dim)

        self.only_crop = cfg.only_crop
        self.FC_layers = nn.Linear(cfg.num_feature_node, 1) if cfg.only_crop else \
                         fc_layers(1+cfg.bbox_num, int(cfg.num_feature_node**0.5))

    def forward(self, im_data, crop_boxes, det_boxes):
        '''
        im_data: (B, 3, H, W)
        crop_boxes: (N, 5) 
                    N indicate crop box number
                    5 indicate [Mos, xmin, ymin, xmax, ymax]
        det_boxes: (D, 5) 
                   D indicate detect box number
        '''
        N = crop_boxes.shape[0]
        D = det_boxes.shape[0]
        H, W = im_data.shape[2], im_data.shape[3]
        
        # conv
        f3,f4,f5 = self.Feat_ext(im_data)
        cat_feat = torch.cat((self.downsample2(f3),f4,0.5*self.upsample2(f5)),1)
        red_feat = self.DimRed(cat_feat)

        # ROI/D Align
        RoI_feat_crop = self.RoIAlign(red_feat, crop_boxes).unsqueeze(2)
        RoD_feat_crop = self.RoDAlign(red_feat, crop_boxes).unsqueeze(2)
        crop_feat = torch.cat((RoI_feat_crop, RoD_feat_crop), 2).view(N, self.reddim, -1)
        RoI_feat_det = self.RoIAlign(red_feat, det_boxes).unsqueeze(2)
        RoD_feat_det = self.RoDAlign(red_feat, det_boxes).unsqueeze(2)
        det_feat = torch.cat((RoI_feat_det, RoD_feat_det), 2).view(D, self.reddim, -1)
        det_feat = det_feat.view(D*self.reddim, -1).repeat(N, 1, 1)
        node_feat = torch.cat((crop_feat, det_feat), 1)

        # Graph Construction
        node_feat = node_feat.view(N, D+1, -1)
        node_feat = self.node_emb(node_feat)
        box_in_flat = torch.cat((crop_boxes.unsqueeze(1), det_boxes.repeat(N, 1, 1)), 1)
        pos_rela, app_rela = self.GraphConstructor(node_feat, box_in_flat, H, W)
        attn_rela = [pos_rela, app_rela]

        # Graph Attention
        final_feat = self.GraphTransformer(node_feat, attn_rela)
        n, d, feat = final_feat.shape

        final_feat = final_feat[:, 0, :].view(n, 1, 1, feat) if self.only_crop else \
                     final_feat.view(n, d, int(feat**0.5), int(feat**0.5))

        prediction = self.FC_layers(final_feat)
        return prediction


    def _init_weights(self):
        print('Initializing weights...')
        self.DimRed.apply(weights_init)
        self.FC_layers.apply(weights_init)
    
            