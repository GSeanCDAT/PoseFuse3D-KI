import torch

from wan.pipelines.pipeline_wan_i2v_interpolation import WanImageToVideoInterpolationPipeline
from wan.models.transformer_with_condition_FLF2V import WanTransformer3DConditionModel as WanTransformer3DConditionModel_FLF2V
from wan.models.transformer_with_condition import WanTransformer3DConditionModel
from wan.models.condition_module import MultiCtrlModel
from transformers import CLIPVisionModel
from diffusers import AutoencoderKLWan
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPImageProcessor
from safetensors import safe_open
import os

from transformers import AutoTokenizer, UMT5EncoderModel
from wan.models.wrapper import TextWrapper
from wan.pipelines.pipeline_wan_i2v_interpolation import NEGATIVE_PROMPT_TEMPLATE

from tqdm import tqdm

from PIL import Image
from wan.utils.validation import save_videos_from_pil
import numpy as np

import argparse
from accelerate.utils import set_seed

set_seed(1024)

import json
import copy
from wan.utils.human_models.human_models import smpl_x
import torchvision.transforms as transforms



smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
        'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3), 'expression': (-1, 10)}

def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)

def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255.

def read_saftetensor(file_path):
    # Dictionary to store the loaded tensors
    tensors = {}

    # Open the .safetensors file and read its contents
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors

def w_concat_pil(img_pil_lst):
    # horizontally concat PIL images 
    # NOTE(ZSH): assume all images are of same size
    W, H = img_pil_lst[0].size
    num_img = len(img_pil_lst)
    new_width = num_img * W
    new_image = Image.new("RGB", (new_width, H), color=0)
    for img_idx, img in enumerate(img_pil_lst):
        new_image.paste(img, (W * img_idx, 0))  
    
    return new_image

def h_concat_pil(img_pil_lst):
    # horizontally concat PIL images 
    # NOTE(ZSH): assume all images are of same size
    W, H = img_pil_lst[0].size
    num_img = len(img_pil_lst)
    new_height = num_img * H
    new_image = Image.new("RGB", (W, new_height), color=0)
    for img_idx, img in enumerate(img_pil_lst):
        new_image.paste(img, (0, H * img_idx, ))  
    
    return new_image

def project_vertices(vertices: np.ndarray,
                            camera_pose: np.ndarray,
                            fx: float, fy: float,
                            cx: float, cy: float):
    """
    Batched matrix-multiply projection of 3D vertices to 2D.

    Args:
      vertices       (Nx3 array): 3D points in world coords.
      camera_pose    (4x4 array): World→camera transform.
      fx, fy, cx, cy (floats):    Intrinsic parameters.

    Returns:
      projected (Nx2 array): Pixel coordinates.
      depth     (N, )     : Z values in camera coords.
    """
    N = vertices.shape[0]
    K = np.array([[fx,  0, cx, 0],
                  [ 0, fy, cy, 0],
                  [ 0,  0,  1, 0]])
    
    ones = np.ones((N, 1), dtype=vertices.dtype)
    vertices_h = np.hstack([vertices, ones]).T    # 4×N

    # 3. Transform to camera coords: (4×4)×(4×N) → (4×N)
    verts_cam_h = camera_pose @ vertices_h       # 4×N 

    # 4. Apply intrinsics: (3×4)×(4×N) → (3×N)
    pix_h = K @ verts_cam_h                      # 3×N 

    # 5. Normalize homogeneous coords
    x = pix_h[0, :] / pix_h[2, :]
    y = pix_h[1, :] / pix_h[2, :]
    projected = np.vstack([x, y]).T              # N×2

    # 6. Depth is z in camera coords (before projection)
    depth = verts_cam_h[2, :]                    # (N,)

    # Ensure no negative depth
    assert np.all(depth > 0), "Some points are behind the camera"

    return projected

def get_smplx_info(pose_img_path, body_model, aug_info=[1,1,0,0] ,human_ids=[1]):
    w_ratio, h_ratio, crop_xmin, crop_ymin = aug_info
    smplx_info_path = pose_img_path.replace('/pose/', '/smplx_ann/')
    smplx_info_basedir = os.path.dirname(smplx_info_path)
    smplx_info_basename = os.path.basename(smplx_info_path)[:-4]

    meta_dir = os.path.join(smplx_info_basedir, 'meta')
    smplxann_dir = os.path.join(smplx_info_basedir, 'smplx')

    metas = []
    for hid in human_ids:
            with open(os.path.join(meta_dir,
                            smplx_info_basename+'_'+str(hid)+'.json'), 'r') as file:
                    metas.append(json.load(file))

    smplx_anns = [np.load(
    os.path.join(smplxann_dir,
                    smplx_info_basename+'_'+str(hid)+'.npz')
    ) for hid in human_ids]

    all_points = []
    all_joints = []
    all_joint_poses = []

    for anno, meta in zip(smplx_anns, metas):
        intersect_key = list(set(anno.keys()) & set(smplx_shape.keys()))
        body_model_param_tensor = {key: torch.tensor(
                np.array(anno[key]).reshape(smplx_shape[key]), device=torch.device('cpu'), dtype=torch.float32)
                        for key in intersect_key if len(anno[key]) > 0}
        output = body_model(**body_model_param_tensor, 
                                    return_verts=True, 
                                    return_full_pose=True)
        # use full point cloud & joints
        vertices = output['vertices'].detach().cpu().numpy().squeeze()
        joint_pose = output['full_pose'].detach().cpu().numpy().squeeze().reshape(55,3)
        joints = output['joints'].detach().cpu().numpy().squeeze()


        focal_length = meta['focal']
        principal_point = meta['princpt']
        fx=focal_length[0]
        fy=focal_length[1]
        cx=principal_point[0] 
        cy=principal_point[1]
        fx, fy, cx, cy = fx * w_ratio, fy * h_ratio, (cx-crop_xmin) * w_ratio, (cy-crop_ymin) * h_ratio
        
        camera_pose = np.eye(4)
        proj_vertices = project_vertices(vertices, camera_pose, fx, fy, cx, cy)
        vertices = np.concatenate([vertices, proj_vertices], axis=-1)
        proj_joints = project_vertices(joints, camera_pose, fx, fy, cx, cy)
        joints = np.concatenate([joints, proj_joints], axis=-1)

        all_points.append(vertices)
        all_joints.append(joints)
        all_joint_poses.append(joint_pose)
    all_points = np.stack(all_points)
    all_joints = np.stack(all_joints)
    all_joint_poses = np.stack(all_joint_poses)

    # n_HUMAN, j/P, 3
    return {'points': all_points, 
            'joints': all_joints, 
            'joint_poses': all_joint_poses}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_name", type=str, default="")
    parser.add_argument("--inference_dtype", type=str, default="bf16")

    parser.add_argument("--example_folder_path", type=str, default="examples/example1")

    parser.add_argument(
        "--use_flf2v",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--normal_cond",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--depth_cond",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--smplx_info",
        action="store_true",
        default=False,
    )


    args = parser.parse_args()

    if args.inference_dtype == 'bf16':
        weight_dtype = torch.bfloat16
    elif args.inference_dtype == 'fp32':
        weight_dtype = torch.float32
    else:
        raise NotImplementedError

    checkpoint_path = 'ckpt/' + args.checkpoint_name

    if args.use_flf2v:
        model_id = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"
    else:
        model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=weight_dtype)

    if os.path.exists("empty_prompt_embedding.pt") and \
        os.path.exists("negative_prompt_embedding.pt"):
        prompt_embeds = torch.load("empty_prompt_embedding.pt", weights_only=True).to('cuda')
        negative_prompt_embeds = torch.load("negative_prompt_embedding.pt", weights_only=True).to('cuda')
        # import pdb;pdb.set_trace()
    else:
        text_encoder = UMT5EncoderModel.from_pretrained(model_id, 
                                                        subfolder="text_encoder")
        tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                                        subfolder="tokenizer")
        text_wrapper = TextWrapper(text_encoder=text_encoder,
                                    tokenizer=tokenizer).to('cuda')
        text_wrapper.requires_grad_(False)

        with torch.no_grad():
            prompt_embeds = text_wrapper.encode_prompt(prompt='', dtype=torch.float32, device='cuda')
            negative_prompt_embeds = text_wrapper.encode_prompt(prompt=NEGATIVE_PROMPT_TEMPLATE, dtype=torch.float32, device='cuda')
            text_wrapper = text_wrapper.to('cpu')
            import pdb;pdb.set_trace()
            torch.save(prompt_embeds.cpu(), "empty_prompt_embedding.pt")
            torch.save(negative_prompt_embeds.cpu(), "negative_prompt_embedding.pt")
    # prompt_embed = torch.load('empty_prompt_embedding.pt', weights_only=True).to('cuda')
    # negative_prompt_embed = torch.load('negative_prompt_embedding.pt', weights_only=True).to('cuda')

    shift = 3.0

    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=weight_dtype
    )

    if args.use_flf2v:
        transformer = WanTransformer3DConditionModel_FLF2V.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=weight_dtype
        )
    else:
        transformer = WanTransformer3DConditionModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=weight_dtype
        )

    transformer.load_lora_adapter(os.path.join(checkpoint_path, 'pytorch_lora_weights.safetensors'))

    control_model = MultiCtrlModel(num_transformer_layers=2,
                                    win_len=[5,5],
                                    depth_cond=args.depth_cond, 
                                    normal_cond=args.normal_cond, 
                                    smplx_cond=args.smplx_info, 
                                    )
    control_model_ckpt = read_saftetensor(os.path.join(checkpoint_path, 'control_model.safetensors'))
    control_model.load_state_dict(control_model_ckpt, strict=True)
    control_model = control_model.to('cuda')

    image_processor = CLIPImageProcessor.from_pretrained(model_id, 
                                                    subfolder="image_processor")

    noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)

    pipe = WanImageToVideoInterpolationPipeline(
            vae=vae, 
            image_encoder=image_encoder, 
            image_processor=image_processor,
            scheduler=noise_scheduler,
            transformer=transformer,
            control_model=control_model,
            flf2v_style=args.use_flf2v,
    )
    pipe.to(device='cuda',dtype=weight_dtype)



    print("Start inference ...")

    target_size = (1024,576)
    hr, wr = 576/1080, 1024/1920

    body_model = copy.deepcopy(smpl_x.layer['neutral']).to(torch.device('cpu'))

    pixel_transforms = transforms.Compose([
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])

    example_folder_path = args.example_folder_path
    save_dir = f'{example_folder_path}/output'

    os.makedirs(save_dir, exist_ok=True)
    first_image = Image.open(f'{example_folder_path}/frames/frame_0000.png').resize(target_size)
    last_image = Image.open(f'{example_folder_path}/frames/frame_0024.png').resize(target_size)

    all_pose_impaths = [f'{example_folder_path}/smplx_ann/img_cs/frame_{idx:04d}.png' for idx in range(25)]
    rendered_3d_poses = np.array([pil_image_to_numpy(Image.open(f).resize(target_size)) for f in all_pose_impaths])
    all_conds_3d = pixel_transforms(numpy_to_pt(rendered_3d_poses))

    all_pose_impaths2d = [f'{example_folder_path}/pose/frame_{idx:04d}.png' for idx in range(25)]
    poses2d = np.array([pil_image_to_numpy(Image.open(f).resize(target_size)) for f in all_pose_impaths2d])
    all_conds_2d = pixel_transforms(numpy_to_pt(poses2d))

    smplx_info = {'points': [], 'joints': [], 'joint_poses': []}
    for pp in all_pose_impaths2d:
        cur_smplx_info = get_smplx_info(pp, body_model, aug_info=[wr, hr, 0, 0])
        for key in cur_smplx_info:
            smplx_info[key].append(np.array(cur_smplx_info[key]))
    for k in smplx_info:
        smplx_info[k] = torch.from_numpy(np.stack(smplx_info[k])).to(torch.float32)

    with torch.no_grad() and torch.autocast("cuda", dtype=weight_dtype):

        all_conds_2d = all_conds_2d.to("cuda")
        all_conds_3d = all_conds_3d.to("cuda")

        points = smplx_info['points'].to("cuda") 
        joints = smplx_info['joints'].to("cuda") 
        joint_poses = smplx_info['joint_poses'].to("cuda") 
        num_frames, _, height, width = all_conds_2d.shape

        video = pipe(
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
                points=points, joints=joints, joint_poses=joint_poses
                ).frames[0]

        # collect videos from all process to process zero
        def process_tensor_to_image(img_tensor):
            return [Image.fromarray(((i*0.5 + 0.5).permute(1,2,0).squeeze().numpy()*255).astype(np.uint8)) for i in img_tensor]
        def pipeout_array_to_image(pipe_array):
            return [Image.fromarray((arr*255).astype(np.uint8)) for arr in pipe_array]

        # import pdb;pdb.set_trace()
        video_pil_lst = [pipeout_array_to_image(video), 
                        process_tensor_to_image(all_conds_2d.cpu()), 
                        process_tensor_to_image(all_conds_3d.cpu()),
                        ]
        concat_list = []
        for cur_id in range(len(video_pil_lst[0])):
            os.makedirs(f'{save_dir}/images/', exist_ok=True)
            video_pil_lst[0][cur_id].save(f'{save_dir}/images/{cur_id:02d}.png')
        save_videos_from_pil(video_pil_lst[0], f'{save_dir}/interpolated_video.mp4', fps=4)
        save_videos_from_pil(video_pil_lst[1], f'{save_dir}/2d_pose_video.mp4', fps=4)
        save_videos_from_pil(video_pil_lst[2], f'{save_dir}/3d_pose_video.mp4', fps=4)
