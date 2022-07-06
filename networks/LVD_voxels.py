import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torch
import torch.nn.functional as F


class Network(NetworkBase):
    def __init__(self, hidden_dim=512, output_dim=6890, input_dim=1, b_min = np.array([-1.2, -1.4, -1.2]), b_max = np.array([1.2, 1.3, 1.2])):
        super(Network, self).__init__()
        self._name = 'voxel_encoder'

        self.b_min = torch.FloatTensor(b_min).cuda()
        self.b_max = torch.FloatTensor(b_max).cuda()
        self.bb = self.b_max - self.b_min

        self.conv_1 = nn.utils.weight_norm(nn.Conv3d(input_dim, 32, 3, stride=2, padding=1))  # out: 32
        self.conv_1_1 = nn.utils.weight_norm(nn.Conv3d(32, 32, 3, padding=1))  # out: 32
        self.conv_2 = nn.utils.weight_norm(nn.Conv3d(32, 64, 3, padding=1))  # out: 16
        self.conv_2_1 = nn.utils.weight_norm(nn.Conv3d(64, 64, 3, padding=1))  # out: 16
        self.conv_3 = nn.utils.weight_norm(nn.Conv3d(64, 96, 3, padding=1))  # out: 8
        self.conv_3_1 = nn.utils.weight_norm(nn.Conv3d(96, 96, 3, padding=1))  # out: 8
        self.conv_4 = nn.utils.weight_norm(nn.Conv3d(96, 128, 3, padding=1))  # out: 8
        self.conv_4_1 = nn.utils.weight_norm(nn.Conv3d(128, 128, 3, padding=1))  # out: 8
        self.conv_5 = nn.utils.weight_norm(nn.Conv3d(128, 128, 3, padding=1))  # out: 8
        self.conv_5_1 = nn.utils.weight_norm(nn.Conv3d(128, 128, 3, padding=1))  # out: 8
        self.conv_6 = nn.utils.weight_norm(nn.Conv3d(128, 128, 3, padding=1))  # out: 8
        self.conv_6_1 = nn.utils.weight_norm(nn.Conv3d(128, 128, 3, padding=1))  # out: 8
        self.conv_7 = nn.utils.weight_norm(nn.Conv3d(128, 128, 3, padding=1))  # out: 8
        self.conv_7_1 = nn.utils.weight_norm(nn.Conv3d(128, 128, 3, padding=1))  # out: 8

        feature_size = (3 + input_dim + 32 + 64 + 96 + 128 + 128 + 128 + 128)
        self.fc_0 = nn.utils.weight_norm(nn.Conv1d(feature_size, hidden_dim, 1))
        self.fc_1 = nn.utils.weight_norm(nn.Conv1d(hidden_dim, hidden_dim*2, 1))
        self.fc_2 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
        self.fc_3 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
        self.fc_4 = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, hidden_dim*2, 1))
        self.fc_out = nn.utils.weight_norm(nn.Conv1d(hidden_dim*2, output_dim, 1))
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x):
        features0 = x
        x = self.actvn(self.conv_1(x))
        x = self.actvn(self.conv_1_1(x))
        features1 = x
        x = self.maxpool(x)

        x = self.actvn(self.conv_2(x))
        x = self.actvn(self.conv_2_1(x))
        features2 = x
        x = self.maxpool(x)

        x = self.actvn(self.conv_3(x))
        x = self.actvn(self.conv_3_1(x))
        features3 = x
        x = self.maxpool(x)

        x = self.actvn(self.conv_4(x))
        x = self.actvn(self.conv_4_1(x))
        features4 = x
        x = self.maxpool(x)

        x = self.actvn(self.conv_5(x))
        x = self.actvn(self.conv_5_1(x))
        features5 = x
        x = self.maxpool(x)

        x = self.actvn(self.conv_6(x))
        x = self.actvn(self.conv_6_1(x))
        features6 = x
        x = self.maxpool(x)

        x = self.actvn(self.conv_7(x))
        x = self.actvn(self.conv_7_1(x))
        features7 = x

        self.features = [features0, features1, features2, features3, features4, features5, features6, features7]

    def query(self, p):
        _B, _numpoints, _ = p.shape

        normalized_p = (p - self.b_min)/self.bb*2 - 1
        point_features = normalized_p.permute(0, 2, 1)
        for j, feat in enumerate(self.features):
            interpolation = F.grid_sample(feat, normalized_p.unsqueeze(1).unsqueeze(1), align_corners=False).squeeze(2).squeeze(2)
            point_features = torch.cat((point_features, interpolation), 1)

        point_features = self.actvn(self.fc_0(point_features))
        point_features = self.actvn(self.fc_1(point_features))
        point_features = self.actvn(self.fc_2(point_features))
        point_features = self.actvn(self.fc_3(point_features))
        point_features = self.actvn(self.fc_4(point_features))
        point_features = self.fc_out(point_features)

        return point_features
