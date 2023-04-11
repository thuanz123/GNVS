# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import PIL
import imageio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src import dnnlib
from src.torch_utils import distributed as dist
from src.torch_utils import training_stats
from src.torch_utils import misc
from src.sample.sample import Sample

def plot_grad_flow(named_parameters, saved_path):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and ("module.renderer" in n):
            layers.append(n)
            print(n, p.grad.detach().abs().mean().cpu().numpy())
            ave_grads.append(p.grad.detach().abs().mean().cpu().numpy())
            max_grads.append(p.grad.detach().abs().max().cpu().numpy())

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(saved_path)

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def convert_torch_to_np(images, value_range=[-1, 1]):
    images = images.permute(0, 2, 3, 1).cpu().numpy()   # batch, c, h, w -> batch, h, w, c
    lo, hi = value_range

    images = np.asarray(images, dtype=np.float32)

    images = (images - lo) * (255 / (hi - lo))
    images = np.rint(images).clip(0, 255).astype(np.uint8)

    return images

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_train_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_train_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    

    # Load test dataset
    if dist.get_rank() == 0:
        from src.dataset.SRNDataset import TestSRNDataset

        config_path = '/lustre/scratch/client/vinai/users/tungdt33/GNVS/training-runs/00005-train_ShapeNet-uncond-ddpmpp-edm-gpus4-batch64-fp32/training_options.json'

        sampler = Sample(config_path, device)


        dist.print0('Loading test dataset...')
        test_dataset_obj = TestSRNDataset(dataset_train_kwargs["path"], stage="test", image_size=(128, 128), world_scale=1.0)
        test_dataset_sampler = misc.InfiniteSampler(dataset=test_dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
        test_dataset_iterator = iter(torch.utils.data.DataLoader(dataset=test_dataset_obj, sampler=test_dataset_sampler, batch_size=1, **data_loader_kwargs))
    
    # Construct network.
    dist.print0('Constructing network...')
    # interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs) # , **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    # if dist.get_rank() == 0:
    #     with torch.no_grad():
    #         images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    #         sigma = torch.ones([batch_gpu], device=device)
    #         labels = torch.zeros([batch_gpu, net.label_dim], device=device)
    #         misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)
            # torch.Size([96, 3, 3, 128, 128]) train_images
            # torch.Size([96, 1, 3, 128, 128]) target_images 
            # torch.Size([96, 1, 64, 64, 8])   target_rays

    
    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                data = next(dataset_iterator)
                train_images = data["train_images"].to(device).to(torch.float32)

                target_rays = data["target_rays"].to(device)
                target_images = data["target_images"].to(device).to(torch.float32) 

                loss, _, _, _, _, _ = loss_fn(net=ddp, train_images=train_images, target_images=target_images, target_rays=target_rays, augment_pipe=augment_pipe)
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        if dist.get_rank() == 0 and cur_nimg % (kimg_per_tick * 1000 * snapshot_ticks) == 0:
            grad_img_path = os.path.join(run_dir, f'grad_{cur_nimg//1000:06d}.png')
            plot_grad_flow(ddp.named_parameters(), grad_img_path)
        
        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')
        

        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            print("Save network snapshot")
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_train_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory
            
            if dist.get_rank() == 0:
                with torch.no_grad():
                    data = next(dataset_iterator)
                    train_images = data["train_images"].to(device).to(torch.float32) 

                    target_rays = data["target_rays"].to(device)
                    target_images = data["target_images"].to(device).to(torch.float32)

                    loss, target_images, D_yn, noises, feature_maps, depth_final = loss_fn(net=ddp, train_images=train_images, target_images=target_images, target_rays=target_rays, augment_pipe=augment_pipe)
                    
                    target_images = target_images.cpu().numpy()
                    D_yn = D_yn.cpu().numpy()
                    loss = loss.cpu().numpy()
                    noises = noises.cpu().numpy()

                    feature_maps = feature_maps.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                    feature_maps -= feature_maps.min(dim=1, keepdim=True)[0]
                    feature_maps /= feature_maps.max(dim=1, keepdim=True)[0]
                    feature_maps = feature_maps.cpu().numpy()

                    depth_final = depth_final.repeat(1, 3, 1, 1)
                    depth_final -= depth_final.min(dim=1, keepdim=True)[0]
                    depth_final /= depth_final.max(dim=1, keepdim=True)[0]
                    depth_final = depth_final.cpu().numpy()


                    save_image_grid(depth_final, os.path.join(run_dir, f'depth_{cur_nimg//1000:06d}.png'), drange=[0, 1], grid_size=(4,4))
                    save_image_grid(feature_maps, os.path.join(run_dir, f'feature_{cur_nimg//1000:06d}.png'), drange=[0, 1], grid_size=(4,4))
                    save_image_grid(target_images, os.path.join(run_dir, f'target_{cur_nimg//1000:06d}.png'), drange=[-1, 1], grid_size=(4,4))
                    save_image_grid(D_yn, os.path.join(run_dir, f'denoises_{cur_nimg//1000:06d}.png'), drange=[-1, 1], grid_size=(4,4))
                    save_image_grid(loss, os.path.join(run_dir, f'loss_{cur_nimg//1000:06d}.png'), drange=[-1, 1], grid_size=(4,4))
                    save_image_grid(noises, os.path.join(run_dir, f'noises_{cur_nimg//1000:06d}.png'), drange=[-1, 1], grid_size=(4,4))

                    data = next(test_dataset_iterator)
                    pred_rgb, target_images, depth_maps, feature_maps = sampler.test_ShapeNet(data, ddp.module)
                    np_pred_rgb = convert_torch_to_np(pred_rgb)
                    np_target_images = convert_torch_to_np(target_images)

                    video_out = imageio.get_writer(os.path.join(run_dir, f'pred_video_{cur_nimg//1000:06d}.mp4'), mode='I', fps=20, codec='libx264')

                    for image in np_pred_rgb:
                        video_out.append_data(image)

                    video_out.close()

                    video_out = imageio.get_writer(os.path.join(run_dir, f'target_video_{cur_nimg//1000:06d}.mp4'), mode='I', fps=20, codec='libx264')

                    for image in np_target_images:
                        video_out.append_data(image)
                        
                    video_out.close()


        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            print("Save full dump of the training state")
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                print("Save full dump of the training state")
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
