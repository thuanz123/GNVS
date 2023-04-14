import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.model.FeatureVolumeEncoder import FeatureVolumeEncoder
from src.model.MLP import MLP_Nerf
from src.util.util import sample_coarse_points, sample_from_3dgrid, composite

class Renderer(nn.Module):
    def __init__(self):
        super().__init__()

        self.volume_encoder = FeatureVolumeEncoder()
        self.mlp_nerf = MLP_Nerf()


    def forward_volume(self, images):
        # for inference purpose
        # b, _, h, w = images.shape
        volumes = self.volume_encoder(images)
        return volumes

    def render_from_volumes(self, batch_rays, volumes, norm_src):
        # batch_rays  batch, sample_points, render_h, render_w, 3
        # volumes     batch, sample_points, h, w, feature_channel

        points, deltas, z_samp = sample_coarse_points(batch_rays)
        sampled_features = sample_from_3dgrid(volumes, points, points, norm_src)

        # Note
        # this function is used with only 1 condition images so we dont need to average the feature map

        sb, sample_points, render_h, render_w, c = sampled_features.shape

        point_features = self.mlp_nerf(sampled_features.reshape(sb, -1 , c))        # super_batch, n_samples * render_h * render_w, 17
        point_features = point_features.reshape(sb, sample_points, render_h, render_w, -1)        # super_batch, n_samples, render_h, render_w, 17
        out = composite(point_features, deltas[:sb,...], z_samp[:sb,...])

        rgb_final = out["rgb_final"]
        depth_final = out["depth_final"]

        return rgb_final, depth_final


    def forward(self, images, cond_rays, target_rays, norm_src):
        # images = data["train_images"]                                               # super_batch, 3, 3, 128, 128
        # target_images = data["target_images"]                                       # super_batch, 1, 3, 128, 128
        # target_rays = data["target_rays"]                                           # super_batch, 1, 1,  64,  64

        sb, b, _, h, w = images.shape
        _, _, ray_h, ray_w, ray_c = cond_rays.shape
        
        mb = np.random.randint(1, 4, (1))[0]                                        # random mini batch
        images = images[:, :mb,...]
        cond_rays = cond_rays[:, :mb,...]

        batch_images = images.reshape(sb*mb, 3, h, w)                               # super_batch* mini_batch, 3, 128, 128
        cond_rays = cond_rays.reshape(sb*mb, ray_h, ray_w, ray_c)                   # super_batch* mini_batch, 64, 64, 8
        
        target_rays = target_rays.repeat(1, mb, 1, 1, 1)                            # super_batch, mini_batch, 64, 64, 8
        batch_rays = target_rays.reshape(sb * mb, 64, 64, 8)                        # super_batch * mini_batch, 64, 64, 8
        volumes = self.volume_encoder(batch_images)

        points, deltas, z_samp = sample_coarse_points(batch_rays)     
        if norm_src:  
            cond_points, _, _ = sample_coarse_points(cond_rays)
        else:
            cond_points = None

        sampled_features = sample_from_3dgrid(volumes, points, cond_points, norm_src)

        sampled_features = sampled_features.reshape(sb, mb, 64, 64, 64, 16)
        sampled_features = sampled_features.mean(dim=1)
        sampled_features.reshape(sb, 64, 64, 64, 16)

        sb, sample_points, render_h, render_w, c = sampled_features.shape

        point_features = self.mlp_nerf(sampled_features.reshape(sb, -1 , c))        # super_batch, n_samples * render_h * render_w, 17
        point_features = point_features.reshape(sb, sample_points, render_h, render_w, -1)        # super_batch, n_samples, render_h, render_w, 17

        out = composite(point_features, deltas[:sb,...], z_samp[:sb,...])


        rgb_final = out["rgb_final"]
        depth_final = out["depth_final"]
        

        return rgb_final, depth_final
