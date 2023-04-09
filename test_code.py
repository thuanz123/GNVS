from src.model.FeatureVolumeEncoder import FeatureVolumeEncoder
import torch

dummy_data = torch.rand((1, 3, 128, 128))
volume_encoder = FeatureVolumeEncoder()

volume = volume_encoder(dummy_data)
print(volume.shape)