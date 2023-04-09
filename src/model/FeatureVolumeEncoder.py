from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead
from src.model.DeepLabV3PlusDecoder import DeepLabV3PlusDecoder
import torch

class FeatureVolumeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = get_encoder(
                    "resnet34",
                    in_channels=3,
                    depth=5,
                    weights="imagenet",
                    output_stride=16,
                )


        self.decoder = DeepLabV3PlusDecoder(
                    encoder_channels=self.encoder.out_channels,
                    out_channels=256,
                    atrous_rates=(12, 24, 36),
                    output_stride=16,
                )


        self.segmentation_head = SegmentationHead(
                    in_channels=self.decoder.out_channels,
                    out_channels=1024,
                    activation=None,
                    kernel_size=1,
                    upsampling=4,
                )

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        volume = self.segmentation_head(decoder_output)
        volume = volume.view(volume.shape[0], 16, 64, volume.shape[-2], volume.shape[-1])
        return volume