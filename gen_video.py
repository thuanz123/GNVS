import torch
from src.torch_utils import misc
from src import dnnlib
import pickle
import copy
from src.dataset.SRNDataset import TrainSRNDataset
import numpy as np
import PIL


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

dataset_path = '/lustre/scratch/client/vinai/users/tungdt33/data/cars'


dataset = TrainSRNDataset(dataset_path, stage="train", image_size=(128, 128), world_scale=1.0, use_labels=False, max_size=None, resolution=None)
data_0 = dataset[2]



resume_pkl = '/lustre/scratch/client/vinai/users/tungdt33/GNVS/training-runs/00005-train_ShapeNet-uncond-ddpmpp-edm-gpus4-batch64-fp32/network-snapshot-003503.pkl'

device = "cuda:0"
network_kwargs = {
                        "model_type": "SongUNet",
                        "embedding_type": "positional",
                        "encoder_type": "standard",
                        "decoder_type": "standard",
                        "channel_mult_noise": 1,
                        "resample_filter": [1, 1],
                        "model_channels": 128,
                        "channel_mult": [1, 1, 2, 2, 2],
                        "img_channels": 19,
                        "img_resolution": 128,
                        "out_channels": 3,
                        "label_dim": 0,
                        "class_name": "src.model.networks.EDMPrecond",
                        "dropout": 0.13,
                        "use_fp16": False}

net = dnnlib.util.construct_class_by_name(**network_kwargs) # , **interface_kwargs) # subclass of torch.nn.Module
net.train().requires_grad_(True).to(device)

if resume_pkl is not None:
    with dnnlib.util.open_url(resume_pkl) as f:
        data = pickle.load(f)

    misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
    del data # conserve memory


with torch.no_grad():
    cond_features = []
    rnd = StackedRandomGenerator(device, [2])

    latents = rnd.randn([1, 3, 128, 128], device=device)

    cond_images = data_0["train_images"].unsqueeze(0).to(device)
    target_rays = data_0["target_rays"].unsqueeze(0).to(device)

    cond_feature, _ = net.renderer(cond_images, target_rays)
    cond_features.append(cond_feature)


    cond_feature_map = torch.cat(cond_features, dim=0)

    cond_feature_map = torch.mean(cond_feature_map, dim=0).unsqueeze(0)

    sigma_min = 0.002
    sigma_max = 80
    num_steps = 25
    rho = 7
    S_churn=0
    S_min=0
    S_max=float('inf')
    S_noise=1
    randn_like=torch.randn_like

    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0


    x_next = latents.to(torch.float64) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(noised_images=x_hat, cond_images=None, target_rays=None, sigma=t_hat, class_labels=None, cond_features=cond_feature_map).to(torch.float64)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(noised_images=x_next, cond_images=None, target_rays=None, sigma=t_next, class_labels=None, cond_features=cond_feature_map).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)


    target_images = data_0["target_images"]


    np_target_images = target_images.squeeze(0).permute(1, 2, 0).cpu().numpy()
    np_x_next = x_next.squeeze(0).permute(1, 2, 0).cpu().numpy()

    lo, hi = [-1, 1]

    np_target_images = np.asarray(np_target_images, dtype=np.float32)

    np_target_images = (np_target_images - lo) * (255 / (hi - lo))
    np_target_images = np.rint(np_target_images).clip(0, 255).astype(np.uint8)


    PIL.Image.fromarray(np_target_images, 'RGB').save("test_target.png")



    np_x_next = np.asarray(np_x_next, dtype=np.float32)

    np_x_next = (np_x_next - lo) * (255 / (hi - lo))
    np_x_next = np.rint(np_x_next).clip(0, 255).astype(np.uint8)


    PIL.Image.fromarray(np_x_next, 'RGB').save("test_sample.png")

# with torch.no_grad():
#     train_images = torch.rand([32, 3, 3, 128, 128], device=device)
#     target_images = torch.rand([32, 3, 128, 128], device=device)
#     target_rays = torch.rand([32, 1, 64, 64, 8], device=device)

#     P_mean = -1.2
#     P_std=1.2
#     sigma_data=0.5
    
#     rnd_normal = torch.randn([target_images.shape[0], 1, 1, 1], device=target_images.device)
#     sigma = (rnd_normal * P_std + P_mean).exp()
#     y, augment_labels = target_images, None
#     n = torch.randn_like(y) * sigma

#     misc.print_module_summary(net, [y+n, train_images, target_rays, sigma], max_nesting=2)
   
