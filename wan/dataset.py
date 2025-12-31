import os, io, csv, math, random
import numpy as np
from einops import rearrange

import torch
import json

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

from copy import deepcopy
from PIL import Image, ImageChops

from wan.utils.kps2pose import DWposeDetector, draw_pose
try:
    from wan.utils.human_models.human_models import smpl_x
except:
    print("Reminder: prepare files for human smplx")

try:
    from utils.render import render_annotation
except:
    print("Reminder: prepare files for rendering")
import copy
import gzip
from packaging import version as pver

smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
        'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3), 'expression': (-1, 10)}

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

def read_frames_by_winodw(
                   root_path1, 
                   root_path2, 
                   clip_name_path1, 
                   clip_name_path2,
                   n_frames, 
                   per_clip_sampling=False):
    # Helper function to read clip names from a file
    def get_clip_paths(root_path, clip_name_path):
        with open(clip_name_path, "r") as file:
            clip_names = [line.strip() for line in file if line.strip()]
        return [os.path.join(root_path, 'frames', clip_name) for clip_name in sorted(clip_names)]
    
    # Retrieve clip paths from both root directories
    clip_paths1 = get_clip_paths(root_path1, clip_name_path1)
    clip_paths2 = get_clip_paths(root_path2, clip_name_path2)

    # Combine all clip paths
    all_clip_paths = clip_paths1 + clip_paths2

    # List to store all sliding windows of frame paths
    all_windows = []
    if per_clip_sampling:
        cur_windows = []
    # Process each clip path
    for clip_path in all_clip_paths:
        # Ensure the path exists
        if not os.path.exists(clip_path):
            print(f"Warning: {clip_path} does not exist.")
            continue

        # List all PNG files in the directory and sort them
        frames = sorted([f for f in os.listdir(clip_path) if f.endswith('.png')])
        if n_frames>len(frames):
            continue
        # Apply sliding window
        for i in range(len(frames) - n_frames + 1):
            window = frames[i:i + n_frames]
            # Prepend the clip path to each frame to get the full path
            window_paths = [os.path.join(clip_path, frame) for frame in window]
            if per_clip_sampling:
                cur_windows.append(window_paths)
            else:
                all_windows.append(window_paths)
        if per_clip_sampling:
            all_windows.append(cur_windows)

    return all_windows

def load_bbox(img_path):
    with open(img_path.replace('frames','/bbox/')[:-4]+'.json', 'r') as f:
        data = json.load(f)
    ids = []
    bbox = []
    mask_height = data["mask_height"] 
    mask_width = data["mask_width"]
    for k in sorted(data['labels'].keys()):
        w =  abs(data['labels'][k]['x2']-data['labels'][k]['x1'])
        h =  abs(data['labels'][k]['y2']-data['labels'][k]['y1']) 
        # skip small bboxes by bbox_thr in pixel
        # if w < 50 * mask_width / 1280 or \
        #     h < 50 * 3 * mask_height / 720:
        #     continue
        if w * h < (50 * mask_width / 1280) * (50 * 3 * mask_height / 720):
            continue
        ids.append(data['labels'][k]['instance_id'])
        bbox.append([data['labels'][k]['x1'], 
                     data['labels'][k]['y1'], 
                     data['labels'][k]['x2'], 
                     data['labels'][k]['y2']])
    return np.array(bbox)

def load_mask(img_path):
    mask_path = img_path.replace('frames','/seg/')[:-4]+'.npy'
    if os.path.basename(os.path.dirname(mask_path)).split('_')[0] != 'clip':
        mask_path = mask_path + '.gz'
        with gzip.open(mask_path, 'rb') as f:
            mask = np.load(f)
    else:
        mask = np.load(mask_path)
    return (mask!=0).astype(np.float32)

def get_all_bboxes(img_paths):
    all_bboxes = np.stack([load_bbox(p) for p in img_paths])
    return all_bboxes

def get_all_masks(img_paths):
    all_masks = np.stack([load_mask(p) for p in img_paths])
    return all_masks

class Camera(object):
    def __init__(self, w2c_mat, fx, fy, cx, cy):
        # fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        # w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # NOTE: c2w @ dirctions
    rays_dxo = torch.linalg.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

class HuamnVideoPoseDataset(Dataset):
    def __init__(
            self,
            root_path1='data/new_pexels_ready', 
            clip_name_path1='data/new_pexels_ready/train_split.txt',
            root_path2='data/sportsslomo_fullbody', 
            clip_name_path2='data/sportsslomo_fullbody/train_split.txt',
            width=224,
            height=224,
            sample_n_frames=9,
            interval_frame=1,
            shuffle=True,
            online_render=False,
            test_mode=False,
            test_subset='',
            out_raw=False,
            per_clip_sampling=False,
            get_smplx_param=False,
            single_path='',
            full_img=False,
            path_prefix='',
            read_pose_img=False,
        ):
        self.full_img = full_img
        self.single_path = single_path

        self.path_prefix = path_prefix
        self.read_pose_img = read_pose_img

        if test_mode:
            print('load fixed test frames')
            if single_path != '':
                test_p1 = single_path
                with open(test_p1, "r") as file:
                    data1 = [line.strip().split(', ') for line in file if line.strip()]
                self.data_path_list = data1
                sample_n_frames = len(self.data_path_list)//interval_frame
            else:
                test_p1 = f'new_sportsslomo_testframes{test_subset}.txt'
                test_p2 = f'new_pexels_testframes{test_subset}.txt'
                with open(test_p1, "r") as file:
                    data1 = [line.strip().split(', ') for line in file if line.strip()]
                with open(test_p2, "r") as file:
                    data2 = [line.strip().split(', ') for line in file if line.strip()]
                self.data_path_list = data1 + data2
                sample_n_frames = len(self.data_path_list)//interval_frame
        else:
            self.data_path_list = read_frames_by_winodw(root_path1, root_path2, 
                                                    clip_name_path1, clip_name_path2, 
                                                    sample_n_frames*interval_frame, 
                                                    per_clip_sampling=per_clip_sampling)

        if shuffle:
            random.shuffle(self.data_path_list)    
        self.length           = len(self.data_path_list)
        self.sample_n_frames  = sample_n_frames
        self.width            = width
        self.height           = height
        self.interval_frame   = interval_frame
        self.pose_reader_2d   = DWposeDetector()
        self.online_render    = online_render
        self.test_mode        = test_mode
        self.per_clip_sampling= per_clip_sampling
        self.get_smplx_param  = get_smplx_param

        if self.get_smplx_param:
            self.body_model = copy.deepcopy(smpl_x.layer['neutral']).to(torch.device('cpu'))

        if online_render:
            self.smplx_model      = smpl_x.layer['neutral']
        sample_size           = (height, width)
        self.resize_crop_transform = BboxGuidedRandomCrop(sample_size, test_mode=test_mode)
        self.pixel_transforms = transforms.Compose([
            transforms.Resize((height, width), antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.out_raw = out_raw
        assert not out_raw
        if out_raw:
            self.pixel_transforms = transforms.Compose([
                transforms.Resize(height, antialias=True),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])

    def draw_2d_poses(self, img_path, img_shape):
        if self.read_pose_img:
            pose_img_path = img_path.replace('/frames/', '/poses_133/')
            out = pil_image_to_numpy(Image.open(pose_img_path))
        else:
            pose_path = img_path.replace('/frames/', '/poses_133/')[:-4]+'.json'
            pose = self.pose_reader_2d(pose_path, (img_shape[0], img_shape[1], 3))
            out = draw_pose(pose)
        return out

    def render_3d_poses(self, img_path):
        smplx_info_path = img_path.replace('/frames/', '/smplx_ann/')
        smplx_info_basedir = os.path.dirname(smplx_info_path)
        smplx_info_basename = os.path.basename(smplx_info_path)[:-4]

        rendered_dir = os.path.join(smplx_info_basedir, 'img_cs')
        rendered_path = os.path.join(rendered_dir, 
                                smplx_info_basename+'.png')
        return pil_image_to_numpy(Image.open(rendered_path))


    def get_smplx_info(self, img_path, aug_info):
        w_ratio, h_ratio, crop_xmin, crop_ymin = aug_info
        smplx_info_path = img_path.replace('/frames/', '/smplx_ann/')
        smplx_info_basedir = os.path.dirname(smplx_info_path)
        smplx_info_basename = os.path.basename(smplx_info_path)[:-4]

        meta_dir = os.path.join(smplx_info_basedir, 'meta')
        smplxann_dir = os.path.join(smplx_info_basedir, 'smplx')
        if self.path_prefix != '':
            human_ids, img_height, img_width  = get_ids(img_path.replace(f'data/{self.path_prefix}/', 'data/'))
            human_ids = [ix+1 for ix,_ in enumerate(human_ids)]
        else:
            human_ids, img_height, img_width  = get_ids(img_path)

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
            output = self.body_model(**body_model_param_tensor, 
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

    def read_camera_batch(self, img_paths):
        clip_path = os.path.dirname(img_paths[0])
        
        all_image_names = sorted([f for f in os.listdir(clip_path) if f.endswith('.png')])
        frame_ids = [all_image_names.index(os.path.basename(ip)) for ip in img_paths]
        cam_path = os.path.join(clip_path.replace('/frames/', '/cam/'), 'camera.npy')
        all_cam = np.load(cam_path, allow_pickle=True).item()
        w2c_r = all_cam['world_cam_R'][frame_ids]
        w2c_t = all_cam['world_cam_T'][frame_ids][:, :, None]
        w2cs = np.concatenate([w2c_r, w2c_t], axis=-1)
        fx = fy = all_cam['img_focal']
        cx, cy = all_cam['img_center']
        
        return [Camera(w2c, fx, fy, cx, cy) for w2c in w2cs]

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        cam_to_origin = 0
        # source_cam_c2w = abs_c2ws[0]
        # if self.zero_t_first_frame:
        #     cam_to_origin = 0
        # else:
        #     cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses
    
    def get_batch(self, idx, win_idx=None):
        if win_idx is None:
            cur_frames = self.data_path_list[idx][::self.interval_frame]
        else:
            cur_frames = self.data_path_list[idx][win_idx][::self.interval_frame]
        
        if self.path_prefix != '':
            cur_frames = [f.replace('data/', f'data/{self.path_prefix}/') for f in cur_frames]

        images = np.array([pil_image_to_numpy(Image.open(f)) for f in cur_frames])
        ori_img_shape = images[0].shape
        drawn_2d_poses = np.array([self.draw_2d_poses(f, 
                                           ori_img_shape 
                                           ) for f in cur_frames])
        # if self.single_path != '' or self.path_prefix != '':
        #     cam_params = None
        # else:
        #     cam_params = self.read_camera_batch(cur_frames)
        cam_params = None
        frames = numpy_to_pt(images)
        conds_2d = numpy_to_pt(drawn_2d_poses)

        rendered_3d_poses = np.array([self.render_3d_poses(f) for f in cur_frames])
        # if self.single_path != '' or self.path_prefix != '':
        #     #TODO: change this later
        #     conds_3d = {'conds_3d': numpy_to_pt(rendered_3d_poses), 
        #                 'normal': numpy_to_pt(rendered_3d_poses), 
        #                 'depth': numpy_to_pt(rendered_3d_poses) }
        # else:
        #     conds_3d = {'conds_3d': numpy_to_pt(rendered_3d_poses), 
        #                 'normal': numpy_to_pt(np.array([pil_image_to_numpy(Image.open(f.replace('/frames/', '/smplx_normal_img/'))) for f in cur_frames])), 
        #                 'depth': numpy_to_pt(np.array([pil_image_to_numpy(Image.open(f.replace('/frames/', '/smplx_depth_img/'))) for f in cur_frames])) }

        conds_3d = {'conds_3d': numpy_to_pt(rendered_3d_poses), 
                    'normal': numpy_to_pt(rendered_3d_poses), 
                    'depth': numpy_to_pt(rendered_3d_poses) }

        return frames, conds_2d, conds_3d, cam_params

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.per_clip_sampling:
            win_idx = random.randint(0, len(self.data_path_list[idx])-1)
            raw_frames, raw_conds_2d, raw_conds_3d, cam_params = self.get_batch(idx, win_idx=win_idx)
            # while True:
            #     try:
            #         raw_frames, raw_conds_2d, raw_conds_3d = self.get_batch(idx, win_idx=win_idx)
            #         break
            #     except:
            #         print('trying another win_idx for:')
            #         print(self.data_path_list[idx][win_idx])
            #         win_idx = random.randint(0, len(self.data_path_list[idx])-1)
        else:
            raw_frames, raw_conds_2d, raw_conds_3d, cam_params = self.get_batch(idx)

        if self.out_raw:
            frames = self.pixel_transforms(raw_frames)
            conds_2d = self.pixel_transforms(raw_conds_2d)
            conds_3d = self.pixel_transforms(raw_conds_3d)
            sample = dict(
                frames = frames, 
                conds_2d = conds_2d,
                conds_3d = conds_3d, 
                name = os.path.basename(os.path.dirname(self.data_path_list[idx][0]))
            )
            return sample
        if self.per_clip_sampling:
            all_bboxes = torch.from_numpy(get_all_bboxes(self.data_path_list[idx][win_idx][::self.interval_frame]))
            all_masks = torch.from_numpy(get_all_masks(self.data_path_list[idx][win_idx][::self.interval_frame]))
        else:
            all_bboxes = torch.from_numpy(get_all_bboxes(self.data_path_list[idx][::self.interval_frame]))
            all_masks = torch.from_numpy(get_all_masks(self.data_path_list[idx][::self.interval_frame]))

        target_ratio = self.height /self.width 
        cropped_ratio = -1
        while not (cropped_ratio < 1.1*target_ratio and cropped_ratio > 0.9*target_ratio):
            if cropped_ratio!=-1:
                if self.per_clip_sampling:
                    print(self.data_path_list[idx][win_idx][::self.interval_frame][0])
                else:
                    print(self.data_path_list[idx][::self.interval_frame][0])
            if self.full_img and self.single_path != '':
                frames, conds_2d, conds_3d, bboxes, masks = raw_frames, raw_conds_2d, raw_conds_3d, all_bboxes, all_masks
                aug_info = [1,1,0,0]
            else:
                frames, conds_2d, conds_3d, bboxes, masks, aug_info, intrinsics = self.resize_crop_transform(raw_frames, 
                                                                        raw_conds_2d, 
                                                                        raw_conds_3d, 
                                                                        all_bboxes, 
                                                                        all_masks, 
                                                                        cam_params=cam_params)
            # frames, conds_2d, conds_3d, bboxes, masks, aug_info, intrinsics = self.resize_crop_transform(raw_frames, 
            #                                                         raw_conds_2d, 
            #                                                         raw_conds_3d, 
            #                                                         all_bboxes, 
            #                                                         all_masks,
            #                                                         cam_params=cam_params)
            cropped_ratio = frames.shape[-2]/frames.shape[-1]

        if self.get_smplx_param:
            if self.per_clip_sampling:
                cur_frames = self.data_path_list[idx][win_idx][::self.interval_frame]
            else:
                cur_frames = self.data_path_list[idx][::self.interval_frame]

            if self.path_prefix != '':
                cur_frames = [f.replace('data/', f'data/{self.path_prefix}/') for f in cur_frames]
            
            smplx_info = {'points': [], 'joints': [], 'joint_poses': []}
            for f in cur_frames:
                cur_smplx_info = self.get_smplx_info(f, aug_info)
                for key in cur_smplx_info:
                    smplx_info[key].append(np.array(cur_smplx_info[key]))
            for k in smplx_info:
                smplx_info[k] = torch.from_numpy(np.stack(smplx_info[k])).to(torch.float32)

        else:
            smplx_info = None

        if cam_params is not None:
            if self.single_path == '' and self.path_prefix == '':
                intrinsics = torch.as_tensor(intrinsics)[None]  
                c2w_poses = self.get_relative_pose(cam_params)
                c2w = torch.as_tensor(c2w_poses)[None]

                # t,h,w,6 -> t,6,h,w
                plucker_embedding = ray_condition(intrinsics, c2w, self.height, self.width, 
                                                device='cpu')[0].permute(0, 3, 1, 2).contiguous()
        
        frames = self.pixel_transforms(frames)
        conds_2d = self.pixel_transforms(conds_2d)
        conds_3d = {k:self.pixel_transforms(v) for k,v in conds_3d.items()}
        # print([frames.shape, conds_2d.shape, conds_3d.shape])
        if self.test_mode:
            sample = dict(
                frames = frames, 
                conds_2d = conds_2d,
                bboxes = bboxes, 
                masks = masks
            )
        else:
            sample = dict(
                frames = frames, 
                bboxes = bboxes, 
                conds_2d = conds_2d,
            )
        sample.update(conds_3d)
        sample['depth'] = sample['depth'][:,:1]
        if cam_params is not None:
            if self.single_path == '' and self.path_prefix == '':
                sample['cam_embed'] = plucker_embedding
        if smplx_info is not None:
            sample.update(smplx_info)

        return sample


def get_ids(img_path):
    with open(img_path.replace('frames','/bbox/')[:-4]+'.json', 'r') as f:
        data = json.load(f)
    ids = []
    # bbox = []
    mask_height = data["mask_height"] 
    mask_width = data["mask_width"]
    for k in sorted(data['labels'].keys()):
        w =  abs(data['labels'][k]['x2']-data['labels'][k]['x1'])
        h =  abs(data['labels'][k]['y2']-data['labels'][k]['y1']) 
        if w * h < (50 * mask_width / 1280) * (50 * 3 * mask_height / 720):
            continue
        ids.append(data['labels'][k]['instance_id'])
    return ids, mask_height, mask_width


class BboxGuidedRandomCrop(object):
    def __init__(self, target_size, margin=0.2, test_mode=False):
        self.target_size = target_size  # e.g., (height, width)
        self.margin      = margin
        self.test_mode   = test_mode

    def __call__(self, image, cond1, cond2, boxes, masks_to_aug, 
                 cam_params=None, return_aug_info=False):
        # image: tensor of shape (C, H, W)
        # boxes: tensor of shape (N, 4) with (xmin, ymin, xmax, ymax)
        boxes_to_aug = boxes.clone().to(torch.float)
        boxes = boxes.flatten(0,1)
        # 1. Compute union of all boxes
        xmin = boxes[:, 0].min()
        ymin = boxes[:, 1].min()
        xmax = boxes[:, 2].max()
        ymax = boxes[:, 3].max()
        box_h = int(ymax-ymin)
        box_w = int(xmax-xmin)


        # 2. Expand the union by a random margin (as a fraction of image size)
        T, C, H, W = image.shape
        max_scale = min(H/self.target_size[0], 
                        W/self.target_size[1])
        
        min_scale = min(box_h/self.target_size[0], 
                        box_w/self.target_size[1])
        min_scale = max(min_scale, 1)

        if min_scale >= max_scale:
            # print('case1')
            scale = max_scale
            cur_h, cur_w = int(scale*self.target_size[0]), int(scale*self.target_size[1])
            
            if self.test_mode:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                biggest_box_id = torch.argmax(areas).item()
                center_x = (boxes[biggest_box_id, 0] + boxes[biggest_box_id, 2]).mean(0)/2
                center_y = (boxes[biggest_box_id, 1] + boxes[biggest_box_id, 3]).mean(0)/2
            else:
                random_center_id = random.randint(0, len(boxes)-1)
                center_x = (boxes[random_center_id, 0] + boxes[random_center_id, 2])/2
                center_y = (boxes[random_center_id, 1] + boxes[random_center_id, 3])/2

            crop_xmin = max(0, int(center_x - cur_w/2))
            crop_xmax = min(W, int(crop_xmin + cur_w))

            crop_ymin = max(0, int(center_y - cur_h/2))
            crop_ymax = min(H, int(crop_ymin + cur_h))

        else:
            # print('case2')
            # decide crop size
            if self.test_mode:
                if self.target_size[0] > 256:
                    scale = min(0.5*min_scale + 0.5*max_scale, 
                                2 * min_scale)
                else:
                    random_factor=0
                    scale = random_factor*min_scale + (1-random_factor)*max_scale
            else:
                random_factor = torch.rand(1).item()
                scale = random_factor*min_scale + (1-random_factor)*max_scale
            cur_h, cur_w = int(scale*self.target_size[0]), int(scale*self.target_size[1])

            # find crop anchor
            if self.test_mode:
                if self.target_size[0] > 256:
                    center_x = (xmin+xmax)/2
                    if center_x - cur_w/2<0:
                        crop_xmin = 0
                        crop_xmax = int(crop_xmin + cur_w)
                    else:
                        crop_xmax = min(W, int(center_x + cur_w/2))
                        crop_xmin = int(crop_xmax - cur_w)
                else:
                    random_factor=0.5
                    w_residual = min(cur_w-box_w, xmin)
                    min_w_residual = max(0, cur_w-(W-xmin))
                    crop_xmin = max(0, int(xmin - (w_residual*random_factor+min_w_residual*(1-random_factor))))
                    crop_xmax = min(W, int(crop_xmin + cur_w))
            else:
                random_factor = torch.rand(1).item()
                w_residual = min(cur_w-box_w, xmin)
                min_w_residual = max(0, cur_w-(W-xmin))
                crop_xmin = max(0, int(xmin - (w_residual*random_factor+min_w_residual*(1-random_factor))))
                crop_xmax = min(W, int(crop_xmin + cur_w))

            if self.test_mode:
                if self.target_size[0] > 256:
                    center_y = (ymin+ymax)/2
                    if center_y - cur_h/2<0:
                        crop_ymin = 0
                        crop_ymax = int(crop_ymin + cur_h)
                    else:
                        crop_ymax = min(H, int(center_y+cur_h/2))
                        crop_ymin = int(crop_ymax - cur_h)
                else:
                    random_factor=0.5
                    h_residual = min(cur_h-box_h, ymin)
                    min_h_residual = max(0, cur_h-(H-ymin))
                    crop_ymin = max(0, int(ymin - (h_residual*random_factor+min_h_residual*(1-random_factor))))
                    crop_ymax = min(H, int(crop_ymin + cur_h))
            else:
                random_factor = torch.rand(1).item()
                h_residual = min(cur_h-box_h, ymin)
                min_h_residual = max(0, cur_h-(H-ymin))
                crop_ymin = max(0, int(ymin - (h_residual*random_factor+min_h_residual*(1-random_factor))))
                crop_ymax = min(H, int(crop_ymin + cur_h))


        # 3. Crop the image and adjust the boxes
        image = image[..., crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        cond1 = cond1[..., crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        cond2 = {k:v[..., crop_ymin:crop_ymax, crop_xmin:crop_xmax] for k,v in cond2.items()}
        masks_to_aug = masks_to_aug[..., crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        boxes_to_aug[..., [0, 2]] -= crop_xmin
        boxes_to_aug[..., [1, 3]] -= crop_ymin
        boxes_to_aug[..., [0, 2]] = boxes_to_aug[..., [0, 2]].clamp(0, crop_xmax-crop_xmin)
        boxes_to_aug[..., [1, 3]] = boxes_to_aug[..., [1, 3]].clamp(0, crop_ymax-crop_ymin)

        w_ratio = self.target_size[1]/(crop_xmax-crop_xmin)
        h_ratio = self.target_size[0]/(crop_ymax-crop_ymin)
        boxes_to_aug[..., [0, 2]] *= w_ratio
        boxes_to_aug[..., [1, 3]] *= h_ratio

        masks_to_aug = F.interpolate(masks_to_aug.unsqueeze(1), size=self.target_size)

        if cam_params is not None:
            intrinsics = np.asarray([[cam_param.fx * w_ratio,
                                    cam_param.fy * h_ratio,
                                    (cam_param.cx-crop_xmin) * w_ratio,
                                    (cam_param.cy-crop_ymin) * h_ratio]
                                    for cam_param in cam_params], dtype=np.float32)
        else:
            intrinsics = None
        aug_info = [w_ratio, h_ratio, crop_xmin, crop_ymin]


        return image, cond1, cond2, boxes_to_aug, masks_to_aug, aug_info, intrinsics


if __name__ == "__main__":

    def rasterize_points(points, width, height):
        """
        Create a black image of shape (height, width), mark each point white.
        
        Parameters:
            points (list of tuple[float, float]): list of (x, y) coordinates
            width (int): image width (number of columns)
            height (int): image height (number of rows)
        
        Returns:
            PIL.Image: a grayscale image with points in white.
        """
        # Initialize black image
        img_arr = np.zeros((height, width), dtype=np.uint8)
        
        for x_f, y_f in points:
            # Round to nearest integer pixel indices
            x = int(round(x_f))
            y = int(round(y_f))
            # Skip points outside the image bounds
            if 0 <= x < width and 0 <= y < height:
                img_arr[y, x] = 255  # white pixel
        
        return Image.fromarray(img_arr, mode='L')

    dataset = HuamnVideoPoseDataset(
        root_path1 = 'data/new_pexels_ready'  , 
        clip_name_path1 = 'data/new_pexels_ready/train_split.txt',
        root_path2 = 'data/sportsslomo_fullbody',
        clip_name_path2 = 'data/sportsslomo_fullbody/train_split.txt',
        interval_frame=1,
        sample_n_frames=25,
        height=320,
        width=512,
        test_mode=False,
        test_subset='',
        shuffle=False,
        out_raw=False,
        per_clip_sampling=True,
        get_smplx_param=True
    )
    a=dataset[0]
    # print(a['depth'].shape)
    # Image.fromarray(((a['conds_3d'][0].permute(1,2,0).numpy()+1)/2*255).astype(np.uint8)).save('tmp_i.png')
    # rasterize_points(a['points'][0,0,:,-2:].numpy(), 512, 320).save('tmp_p.png')
    import pdb;pdb.set_trace()

