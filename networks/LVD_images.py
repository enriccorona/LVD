import torch.nn as nn
import cv2
import numpy as np
from .networks import NetworkBase
import torchvision
import torch
import torch.nn.functional as F
from .HGFilters import HGFilter
from . import geometry

class Network(NetworkBase):
    def __init__(self, input_point_dimensions=3, input_channels=3, pred_dimensions=6890):
        super(Network, self).__init__()
        self._name = 'DeformationNet'

        self.image_filter = HGFilter(4, 2, input_channels, 256, 'group', 'no_down', False)

        self.fc1 = nn.utils.weight_norm(nn.Conv1d(1024 + input_point_dimensions, 512, kernel_size=1, bias=True))
        self.fc2 = nn.utils.weight_norm(nn.Conv1d(512, 512, kernel_size=1, bias=True))
        self.fc3 = nn.utils.weight_norm(nn.Conv1d(512, pred_dimensions, kernel_size=1, bias=True))

        self.frequencies = [  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.]

    def forward(self, image):
        self.im_feat_list, self.normx = self.image_filter(image)
        return

    def query(self, points):
        # Orthogonal renders are done with scale of 0.2: TODO UPDATE WITH ON-THE-WILD IMAGES WITH WEAK PROJECTION VALS.
        xy = (points/0.2 + 512)/1024
        xy[:,:,1] = 1-xy[:,:,1]
        xy = xy*2 - 1
        points = points/50

        intermediate_preds_list = points.transpose(2, 1)
        for j, im_feat in enumerate(self.im_feat_list):
            point_local_feat_list = [intermediate_preds_list, geometry.index(im_feat, xy)]
            intermediate_preds_list = torch.cat(point_local_feat_list, 1)

        x = F.relu(self.fc1(intermediate_preds_list))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def query_test(self, points):
        # More efficient function for testing:
        xy = (points/0.2 + 512)/1024
        xy[:,:,1] = 1-xy[:,:,1]
        xy = xy*2 - 1
        points = points/50

        intermediate_preds_list = points.transpose(2, 1)
        for j, im_feat in enumerate(self.im_feat_list):
            point_local_feat_list = [intermediate_preds_list, geometry.index(im_feat, xy)]
            intermediate_preds_list = torch.cat(point_local_feat_list, 1)

        x = F.relu(self.fc1(intermediate_preds_list))
        x = F.relu(self.fc2(x))
        if not hasattr(self, 'inference_weights'):
            self.fc3(x) # Run to setup fc3
            self.inference_weights = self.fc3.weight[:,:,0].permute(1,0).reshape(-1,6890,3)
            self.inference_bias = self.fc3.bias.reshape(6890,3)
        x = torch.einsum('fs,fsd->sd', x[0], self.inference_weights) + self.inference_bias

        return x
