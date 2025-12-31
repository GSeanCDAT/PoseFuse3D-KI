# isort: skip_file
import gc
import os
from typing import List, Optional, Union
import copy

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from tqdm import tqdm

import wandb
from wan.utils.communications import all_gather
from wan.utils.parallel_states import (get_sequence_parallel_state, nccl_info)
from wan.constants import normalize_dit_input
from wan.pipelines.pipeline_wan_i2v_interpolation import WanImageToVideoInterpolationPipeline

from PIL import Image
from pathlib import Path
import imageio

def save_videos_from_pil(pil_images, path, fps=24, crf=23):

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if save_fmt == ".mp4":
        with imageio.get_writer(path, fps=fps) as writer:
            for img in pil_images:
                img_array = np.array(img)  # Convert PIL Image to numpy array
                writer.append_data(img_array)

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")

@torch.no_grad()
@torch.autocast("cuda", dtype=torch.bfloat16)
def log_validation(
    args,
    vae,
    image_encoder,
    image_processor,
    tokenizer,
    text_encoder,
    prompt_embeds,
    negative_prompt_embeds,
    main_model,
    val_dataset,
    device,
    weight_dtype,  # TODO
    global_step,
    scheduler_type="euler",
    shift=1.0,
    flf2v_style=False,
):
    # TODO
    print("Running validation....\n")

    vae_spatial_scale_factor = 8
    vae_temporal_scale_factor = 4
    num_channels_latents = 16
    main_model.eval()
    # vae.enable_tiling()
    scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
    transformer = main_model.transformer
    control_model = main_model.control_model
    pipeline = WanImageToVideoInterpolationPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        image_processor=image_processor,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        control_model=control_model,
        wrapped_model=main_model,
        one_step=args.onestep,
        flf2v_style=flf2v_style
    )

    videos = []
    val_idx = global_step%len(val_dataset)
    sample = val_dataset[val_idx]

    all_frames = sample["frames"]
    first_image = Image.fromarray(((sample["frames"][0]*0.5 + 0.5).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
    last_image = Image.fromarray(((sample["frames"][-1]*0.5 + 0.5).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
    all_conds_2d = sample["conds_2d"]
    all_conds_3d = sample["conds_3d"]
    depth = sample["depth"]
    normal = sample["normal"]

    points = sample['points'] if 'points' in sample else None
    joints = sample['joints'] if 'joints' in sample else None
    joint_poses = sample['joint_poses'] if 'joint_poses' in sample else None
    cam_embed = sample['cam_embed']
    num_frames, _, height, width = all_frames.shape

    generator = torch.Generator(device="cpu").manual_seed(12345)
    video = pipeline(
            image1=first_image,
            image2=last_image, 
            control_model_condition = all_conds_2d,
            control_model_condition_add = all_conds_3d,
            prompt='',
            height=height,
            width=width,
            num_frames=num_frames,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            # TODO: adjust the hyperparams
            num_inference_steps = 25,
            guidance_scale = 3.0,
            control_weight=1.0,
            depth=depth,
            normal=normal, 
            points=points, joints=joints, joint_poses=joint_poses,
            cam_embed=cam_embed
            ).frames
    
    
    if nccl_info.rank_within_group == 0:
        videos.append(video[0])
    # collect videos from all process to process zero
    def process_tensor_to_image(img_tensor):
        return [Image.fromarray(((i*0.5 + 0.5).permute(1,2,0).numpy().squeeze()*255).astype(np.uint8)) for i in img_tensor]
    def pipeout_array_to_image(pipe_array):
        return [Image.fromarray((arr*255).astype(np.uint8)) for arr in pipe_array]
    gc.collect()
    torch.cuda.empty_cache()
    # log if main process
    torch.distributed.barrier()
    all_videos = [None for i in range(int(os.getenv("WORLD_SIZE", "1")))]  # remove padded videos
    torch.distributed.all_gather_object(all_videos, videos)


            
    if nccl_info.global_rank == 0:
        sample_name = f"val_{val_idx}"

        videos = [video for videos in all_videos for video in videos]
        video_pil_lst = [pipeout_array_to_image(all_videos[0][0]), 
                        process_tensor_to_image(all_conds_2d), 
                        process_tensor_to_image(all_conds_3d),
                        process_tensor_to_image(all_frames),
                        process_tensor_to_image(depth),
                        process_tensor_to_image(normal)]

        for idx, v in enumerate(video_pil_lst):
            # import pdb;pdb.set_trace()
            save_videos_from_pil(v, f'{args.output_dir}/validation/4fps-{global_step:06d}-{sample_name}-{idx}.mp4', fps=4)
