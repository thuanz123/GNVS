import torch
from src.dataset.SRNDataset import ProcessSRNDataset
import os

device = torch.device('cuda:0')
dataset_path = '/lustre/scratch/client/vinai/users/tungdt33/data/cars'
dataset = ProcessSRNDataset(dataset_path, stage="test", image_size=(128, 128), world_scale=1.0, render_size=(64, 64), use_labels=False, max_size=None, resolution=None)


for i, data in enumerate(dataset):
    print(len(dataset))
    path = data["path"]
    images_path = os.path.join(path, "images_128")
    rays_path = os.path.join(path, "rays_64")

    all_imgs = data["all_imgs"]
    all_rays = data["all_rays"]


    if not os.path.isdir(images_path):
        os.mkdir(images_path)
        os.mkdir(rays_path)

    torch.save(all_imgs, os.path.join(images_path, f"images.pt"))
    torch.save(all_rays, os.path.join(rays_path, f"rays.pt"))
    

        #     images_path = os.path.join(os.path.join(dir_path, "images_128"), "images.pt")
        # rays_path = os.path.join(os.path.join(dir_path, "rays_64"), "rays.pt")
        
        # all_imgs = torch.load(images_path)
        # all_rays = torch.load(rays_path)

        # print("end: ", time.time() - start_time)

        # return all_imgs, all_rays