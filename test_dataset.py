from src.model.Render import Renderer
import torch
from torch.utils.data import DataLoader
from src.model.networks import SongUNet
from src.model.loss import EDMLoss
from src.dataset.SRNDataset import TestSRNDataset
from torch.utils.data import DataLoader

device = torch.device('cuda:0')
dataset_path = '/lustre/scratch/client/vinai/users/tungdt33/data/cars'
dataset = TestSRNDataset(dataset_path, stage="test", image_size=(128, 128), world_scale=1.0)

train_dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=32)

i = 0
# print(len(train_dataloader))
for data in train_dataloader:
    print(i)
    i += 1
