import sys
from torch import nn
from mmfewshot.detection.models.utils.aggregation_layer import AGGREGATORS
from mmcv.runner import BaseModule

import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import cv2


@AGGREGATORS.register_module()
class FeatsPooling(BaseModule):
    def __init__(self):
        super().__init__()
        
    def __call__(self, support_feats, support_gt_labels=None):
        support_feats_m = F.max_pool2d(support_feats, kernel_size=2, stride=2) # max_pool（B，C，H/2，W/2）  # torch.Size([15/5, 1024, 7, 7])
        support_feats_a = F.avg_pool2d(support_feats, kernel_size=2, stride=2) # avg_pool（B，C，H/2，W/2）  # torch.Size([15/5, 1024, 7, 7])
        pooling_feats = support_feats_m - support_feats_a
        return pooling_feats #[15, 1024, 7, 7]

@AGGREGATORS.register_module()
class AwareEnhancement(BaseModule):
    def __init__(self, dim):
        super().__init__()
        k_dim = dim // 4
        self.w_qk = nn.Linear(dim, k_dim, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.))
 
    def forward(self, query_feature, pooling_feats, query_img_metas=None):    
        B, C, H, W = query_feature.shape  # torch.Size([4, 1024, 48, 64])
        b, c, h, w = pooling_feats.shape   # torch.Size([15,1024, 7, 7])4
        q = query_feature.reshape(B, C, H*W).permute(0, 2, 1)   # (B, H*W, 1024)  torch.Size([4, 3072, 1024])
        k = v = pooling_feats.reshape(b, c, h*w).permute(0, 2, 1).reshape(b*h*w, c)  # torch.Size([20*7*7, 1024])
        q = self.w_qk(q)  # (B, H*W, 256)  [4,3072,256]  
        k = self.w_qk(k)  # [20*7*7,256]
        k = k.unsqueeze(0) #   [1, 15*14*14= 2940, 256]
        B, Nt, E = q.shape  # [4,4154,256] 
        attn = torch.bmm(q / math.sqrt(E), k.expand(B, -1, -1).transpose(-2, -1))  # [4,4154,256] X [B=4,256, 2940] = [4,4154, 2940]
        out = torch.matmul(attn.softmax(-1), v)  # [4,4154,15*14*14] X [15*14*14, 1024] = torch.Size([4, 4154, 1024])
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)  # [4,1024,48,64]
        out = query_feature + self.gamma * out
        # pdb.set_trace()
        return out

