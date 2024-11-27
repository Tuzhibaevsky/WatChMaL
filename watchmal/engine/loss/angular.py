import torch
from torch import nn

class CosineLoss(nn.Module):
    def __init__(self, swap_yz=False):
        super(CosineLoss, self).__init__()
        self.swap_yz = swap_yz
    def forward(self, prediction, target):
        # ..., 0: Polar angle       ..., 1: Azimuthal angle
        # Calculate the cos(angular distance) between the two angles
        theta1 = prediction[..., 0]
        theta2 = target[..., 0]
        phi1 = prediction[..., 1]
        phi2 = target[..., 1]
        if not self.swap_yz:
            cos_dist =   torch.cos(theta1) * torch.cos(theta2) + torch.sin(theta1) * torch.sin(theta2) * torch.cos(phi1 - phi2)
        if self.swap_yz:
            cos_dist =   torch.sin(theta1) * torch.sin(theta2) * torch.cos(phi1) * torch.cos(phi2) \
                       + torch.cos(theta1) * torch.sin(theta2) * torch.sin(phi2) \
                       + torch.cos(theta2) * torch.sin(theta1) * torch.sin(phi1)
        # Loss is 1 - cos(angular distance)
        # ~ squared near 0 & 2, linear near 1
        angular_loss = 1 - cos_dist

        return torch.mean(angular_loss)