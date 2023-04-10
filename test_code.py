from src.model.Render import Renderer
import torch
from src.dataset import get_split_dataset
from torch.utils.data import DataLoader
from src.model.networks import SongUNet
from src.model.loss import EDMLoss

device = torch.device('cuda:0')
data_dir = '/lustre/scratch/client/vinai/users/tungdt33/data/cars'
train_set, val_set, test_set = get_split_dataset("srn", data_dir, want_split="all", training=True)


train_dataloader = DataLoader(train_set, batch_size=8, shuffle=True)

net = SongUNet(img_resolution=128,                     # Image resolution at input/output.
                        in_channels=19,                        # Number of color channels at input.
                        out_channels=3,                       # Number of color channels at output.
                        label_dim           = 0,            # Number of class labels, 0 = unconditional.
                        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

                        model_channels      = 128,          # Base multiplier for the number of channels.
                        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
                        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
                        num_blocks          = 4,            # Number of residual blocks per resolution.
                        attn_resolutions    = [16],         # List of resolutions with self-attention.
                        dropout             = 0.10,         # Dropout probability of intermediate activations.
                        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

                        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
                        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
                        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
                        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
                        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
                    ).to(device)

loss_func = EDMLoss()

for data in train_dataloader:
    renderer = Renderer().to(device)

    render_images, depth_final, target_images = renderer(data)
    loss = loss_func(net, render_images, target_images)

    breakpoint()

    


## 
## DDPM

breakpoint()
