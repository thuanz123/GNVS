import torch
from src.torch_utils import misc
from src import dnnlib
import pickle
import copy
import numpy as np
import PIL
from tqdm import tqdm
import json

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


class Sample:
    def __init__(self, config_path, device):
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.num_steps = 25
        self.rho = 7
        self.S_churn = 0
        
        self.S_min = 0
        self.S_max = float('inf')
        self.S_noise = 1
        self.randn_like = torch.randn_like 

        self.net = None
        self.device = device
        self.batch_size = 8
        self.random_seed = np.arange(self.batch_size)
        
        self.rnd = StackedRandomGenerator(self.device, self.random_seed)

        # self.config = json.load(open(config_path, "r"))


    def load_model(self, model_path):
        network_kwargs = self.config["network_kwargs"]
        self.net = dnnlib.util.construct_class_by_name(**network_kwargs) # , **interface_kwargs) # subclass of torch.nn.Module
        self.net.train().requires_grad_(True).to(self.device)

        if model_path is not None:
            with dnnlib.util.open_url(model_path) as f:
                data = pickle.load(f)

            misc.copy_params_and_buffers(src_module=data['ema'], dst_module=self.net, require_all=False)
            del data # conserve memory
    
    def sample(self, cond_feature_map, net, batch_size=8):
        latents = self.rnd.randn([batch_size, 3, 128, 128], device=self.device)
        latents = latents[:cond_feature_map.shape[0],...]
        sigma_min = max(self.sigma_min, net.sigma_min)
        sigma_max = min(self.sigma_max, net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0


        x_next = latents.to(torch.float64) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * self.randn_like(x_cur)

            # Euler step.
            denoised, _, _ = net(noised_images=x_hat, cond_images=None, target_rays=None, sigma=t_hat, class_labels=None, cond_features=cond_feature_map)
            denoised = denoised.to(torch.float64)

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                denoised, _, _ = net(noised_images=x_next, cond_images=None, target_rays=None, sigma=t_next, class_labels=None, cond_features=cond_feature_map)
                denoised = denoised.to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next


    def test_ShapeNet(self, data, net):
        with torch.no_grad():
            cond_images = data["cond_images"].to(self.device)                       #  [1, 3, 128, 128]
            target_images = data["target_images"].to(self.device)                   #  [1, 250, 3, 128, 128]
            target_rays = data["target_rays"].to(self.device)                       #  [1, 250, 64, 64, 8]

            sb, mb, c, h, w = target_images.shape
            sb, mb, render_h, render_w, c_rays = target_rays.shape

            target_images = target_images.reshape(sb*mb, c, h, w)                   #  [250, 3, 128, 128]
            target_rays = target_rays.reshape(sb*mb, render_h, render_w, c_rays)    #  [250, 64, 64, 8]

            cond_volume = net.renderer.forward_volume(cond_images)                  #  [1, 16, 64, 128, 128]            
            

            # 250 image ----> split batch
            feature_maps = []
            depth_maps = []
            pred_rgbs = []


            print("Mem before sampling: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))


            for i in tqdm(range(0, target_images.shape[0], self.batch_size)):
                end = target_images.shape[0] if i + self.batch_size > target_images.shape[0] else i + self.batch_size

                print("Mem in sampling: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
                batch_cond_volume = cond_volume.repeat(end-i, 1, 1, 1, 1)       #  [batch_size, 16, 64, 128, 128]
                feature_map, depth_map = net.renderer.render_from_volumes(batch_rays=target_rays[i:end,...], volumes=batch_cond_volume)
                pred_rgb = self.sample(cond_feature_map=feature_map, net=net, batch_size=self.batch_size)

                feature_maps.append(feature_map)
                depth_maps.append(depth_map)
                pred_rgbs.append(pred_rgb)

            feature_maps = torch.cat(feature_maps, dim=0)
            depth_maps = torch.cat(depth_maps, dim=0)
            pred_rgbs = torch.cat(pred_rgbs, dim=0)

        return pred_rgbs, target_images, depth_maps, feature_maps


