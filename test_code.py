from src.model.Render import Renderer
import torch
from src.dataset.SRNDataset import SRNDataset
from torch.utils.data import DataLoader
from src.util.util import gen_rays, sample_coarse_points

device = torch.device('cuda:0')
data_dir = '/lustre/scratch/client/vinai/users/tungdt33/data/cars'

train_set = SRNDataset(data_dir, stage="train", image_size=(128, 128), world_scale=1.0)

train_dataloader = DataLoader(train_set, batch_size=1, shuffle=False)


for data in train_dataloader:
    images = data["images"].reshape(50, 3, 128, 128)  # (NV, 3, H, W)
    poses = data["poses"].reshape(50, 4, 4)  # (NV, 4, 4)
    focal = data["focal"]
    c = data["c"]

    cam_rays = gen_rays(poses, 64, 64, focal, 0.8, 1.8, c=c)
    points, _, _ = sample_coarse_points(cam_rays)

    torch.save(points, "points.pt")
    torch.save(images, "images.pt")
    torch.save(cam_rays, "cam_rays.pt")
    torch.save(poses, "poses.pt")
    break

    


## 
## DDPM

breakpoint()
