from src.sample.sample import Sample
from src.dataset.SRNDataset import TestSRNDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import imageio

config_path = '/lustre/scratch/client/vinai/users/tungdt33/GNVS/training-runs/00005-train_ShapeNet-uncond-ddpmpp-edm-gpus4-batch64-fp32/training_options.json'

device = "cuda:0"
model_path = '/lustre/scratch/client/vinai/users/tungdt33/GNVS/training-runs/00005-train_ShapeNet-uncond-ddpmpp-edm-gpus4-batch64-fp32/network-snapshot-004254.pkl'
dataset_path = '/lustre/scratch/client/vinai/users/tungdt33/data/cars'

testdataset = TestSRNDataset(dataset_path, stage="test", image_size=(128, 128), world_scale=1.0)
test_dataloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=32)

sampler = Sample(config_path, device)
sampler.load_model(model_path)

def convert_torch_to_np(images, value_range=[-1, 1]):
    images = images.permute(0, 2, 3, 1).cpu().numpy()   # batch, c, h, w -> batch, h, w, c
    lo, hi = value_range

    images = np.asarray(images, dtype=np.float32)

    images = (images - lo) * (255 / (hi - lo))
    images = np.rint(images).clip(0, 255).astype(np.uint8)

    return images

    


for i, data in enumerate(test_dataloader):
    pred_rgb, target_images, depth_maps, feature_maps = sampler.test_ShapeNet(data)
    np_pred_rgb = convert_torch_to_np(pred_rgb)
    np_target_images = convert_torch_to_np(target_images)

    video_out = imageio.get_writer("/lustre/scratch/client/vinai/users/tungdt33/GNVS/videos/test_pred_{}.mp4".format(i), mode='I', fps=20, codec='libx264')

    for image in np_pred_rgb:
        video_out.append_data(image)

    video_out.close()

    video_out = imageio.get_writer("/lustre/scratch/client/vinai/users/tungdt33/GNVS/videos/test_target_{}.mp4".format(i), mode='I', fps=20, codec='libx264')

    for image in np_target_images:
        video_out.append_data(image)
        
    video_out.close()
# model -> batch many image -> batch many volume -> batch mean volume -> render batch target pose -> batch feature map -> 

# testing -> batch image -> batch volume -> batch mean volume -> render batch 1 target pose -> batch feature map



    # feature_maps = torch.mean(feature_maps, dim = 1).unsqueeze(1).repeat(1, 3, 1, 1)

    # feature_maps -= feature_maps.min(dim=1, keepdim=True)[0]
    # feature_maps /= feature_maps.max(dim=1, keepdim=True)[0]
    # np_feature_maps = convert_torch_to_np(feature_maps, value_range=[0, 1])

    # depth_maps = depth_maps.repeat(1, 3, 1, 1)

    # depth_maps -= depth_maps.min(dim=1, keepdim=True)[0]
    # depth_maps /= depth_maps.max(dim=1, keepdim=True)[0]

    # depth_maps = convert_torch_to_np(depth_maps, value_range=[0, 1])

    # video_out = imageio.get_writer("/lustre/scratch/client/vinai/users/tungdt33/GNVS/videos/test_depth_{}.mp4".format(i), mode='I', fps=20, codec='libx264')

    # for image in depth_maps:
    #     video_out.append_data(image)

    # video_out.close()

    # video_out = imageio.get_writer("/lustre/scratch/client/vinai/users/tungdt33/GNVS/videos/test_feat_{}.mp4".format(i), mode='I', fps=20, codec='libx264')

    # for image in np_feature_maps:
    #     video_out.append_data(image)

    # video_out.close()
