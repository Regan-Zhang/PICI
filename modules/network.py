import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network_mae(nn.Module):
    def __init__(self, mae, feature_dim, class_num):
        super(Network_mae, self).__init__()
        self.mae = mae
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.mae.encoder_dim, self.mae.encoder_dim),
            nn.ReLU(),
            nn.Linear(self.mae.encoder_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.mae.encoder_dim, self.mae.encoder_dim),
            nn.ReLU(),
            nn.Linear(self.mae.encoder_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        loss_1, _, _, present_1 = self.mae(x_i)
        loss_2, _, _, present_2 = self.mae(x_j)

        z_i = normalize(self.instance_projector(present_1), dim=1)
        z_j = normalize(self.instance_projector(present_2), dim=1)

        c_i = self.cluster_projector(present_1)
        c_j = self.cluster_projector(present_2)

        return (z_i, z_j, c_i, c_j), loss_1, loss_2

    def forward_cluster(self, x):
        _, _, _, present = self.mae(x)
        c = self.cluster_projector(present)
        c = torch.argmax(c, dim=1)
        return c

    def forward_vis(self, x):
        _, _, _, present = self.mae(x)
        z = normalize(self.instance_projector(present), dim=1)
        c = self.cluster_projector(present)
        c = torch.argmax(c, dim=1)
        return z, c

