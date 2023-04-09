import torch
from torch import nn
from torch.nn import functional as F

from src.model.FeatureVolumeEncoder import FeatureVolumeEncoder
from src.model.MLP import MLP_Nerf
from src.util.util import sample_coarse_points, sample_from_3dgrid, composite

class Renderer(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()

        self.device = device
        self.volume_encoder = FeatureVolumeEncoder()
        self.mlp_nerf = MLP_Nerf()

    def forward(self, data):
        images = data["images"].to(self.device)
        target_rays = data["target_rays"].to(self.device) # super_batch, 3, 128, 128


        sb = 1
        b, _, h, w = images.shape
        batch_images = images.reshape(1, 50, 3, 128, 128)   # super_batch, batch, 3, 128, 128


        batch_images = batch_images.reshape(sb*b, 3, h, w)
        batch_rays = batch_rays.reshape(sb*b, 8, 64, 64)
        
        volumes = self.volume_encoder(batch_images)


        points, deltas, z_samp = sample_coarse_points(target_rays)
        sampled_features = sample_from_3dgrid(volumes, points)

        sampled_features = sampled_features.reshape(sb, b, 64, 64, 64, 16)
        sampled_features = sampled_features.mean(dim=1)
        sampled_features.reshape(sb, 64, 64, 64, 16)

        b, sample_points, h, w, c = sampled_features.shape

        point_features = self.mlp_nerf(sampled_features.reshape(b, -1 , c))
        point_features = point_features.reshape(b, sample_points, h, w, -1)

        out = composite(point_features, deltas, z_samp)
        rgb_final = out["rgb_final"].reshape(sb, b, 3, 128, 128)
        depth_final = out["depth_final"].reshape(sb, b, 1, 128, 128)

        return rgb_final, depth_final
