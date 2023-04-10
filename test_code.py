from src.model.Render import Renderer
import torch
from src.dataset import get_split_dataset
from torch.utils.data import DataLoader



device = torch.device('cuda:0')
data_dir = '/lustre/scratch/client/vinai/users/tungdt33/data/cars'
train_set, val_set, test_set = get_split_dataset("srn", data_dir, want_split="all", training=True)


train_dataloader = DataLoader(train_set, batch_size=8, shuffle=True)

for data in train_dataloader:
    renderer = Renderer().to(device)

    rgb_final, depth_final, target_images = renderer(data)
    breakpoint()


## 
## DDPM

breakpoint()
