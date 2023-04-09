from src.model.Render import Renderer
import torch
from src.dataset import get_split_dataset


device = torch.device('cuda:0')
data_dir = '/lustre/scratch/client/vinai/users/tungdt33/data/cars'
train_set, val_set, test_set = get_split_dataset("srn", data_dir, want_split="all", training=True)


data = train_set[0]

renderer = Renderer().to(device)

out = renderer(data)


## 
## DDPM

breakpoint()
