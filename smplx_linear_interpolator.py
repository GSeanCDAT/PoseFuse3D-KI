import numpy as np
import cv2
import math
import json
import torch
from wan.utils.human_models.human_models import smpl_x
from wan.utils.render import render_annotation
import os
import copy

from tqdm import tqdm

from wan.utils.kps2pose_smplx import DWposeDetector, draw_pose

import shutil

import roma

pose_drawer = DWposeDetector()

with open('./smplx_openpose.json') as f: 
    index_information = json.load(f)
reordered_smplx_idx = index_information['smplx_idxs']

smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
        'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3), 'expression': (-1, 10)}

direct_linear_keys = {'betas': (-1, 10), 'transl': (-1, 3), 'expression': (-1, 10)}


def merge_pose_images(pose_imgs, img_height, img_width):

    merged = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for pose in pose_imgs:
        merged = cv2.addWeighted(merged, 1.0, pose, 1.0, 0)
    return merged

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

def main():
    root = 'examples/example2/'


    img_path1 = f'{root}/frames/frame_0000.png'

    img_path2 = f'{root}/frames/frame_0024.png'
    
    save_path = root
    img_save_path = f'{save_path}/frames/'
    os.makedirs(img_save_path, exist_ok=True)


    smplx_cs_save_path = f'{save_path}/smplx_ann/img_cs'
    smplx_meta_save_path = f'{save_path}/smplx_ann/meta'
    smplx_save_path = f'{save_path}/smplx_ann/smplx'
    smplx_pose_save_path = f'{save_path}/pose/'
    os.makedirs(smplx_pose_save_path, exist_ok=True)
    os.makedirs(smplx_cs_save_path, exist_ok=True)
    os.makedirs(smplx_meta_save_path, exist_ok=True)
    os.makedirs(smplx_save_path, exist_ok=True)

    # human_ids, img_height, img_width  = get_ids(img_path1)
    img_height, img_width = cv2.imread(img_path1).shape[:2]

    def read_smplx_ann(impath):
        smplx_info_path = impath.replace('/frames/', '/smplx_ann/')
        smplx_info_basedir = os.path.dirname(smplx_info_path)
        smplx_info_basename = os.path.basename(smplx_info_path)[:-4]
        meta_dir = os.path.join(smplx_info_basedir, 'meta')
        all_meta_names = os.listdir(meta_dir)
        human_ids = list(set([int(name.split('_')[-1].split('.')[0]) for name in all_meta_names]))
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
        
        # TODO change this soon
        return smplx_anns, metas
        

    body_model = copy.deepcopy(smpl_x.layer['neutral']).to(torch.device('cuda'))
    n_temp = 25
    annos1, metas1 = read_smplx_ann(img_path1)
    annos2, metas2 = read_smplx_ann(img_path2)

    intersect_key = list(set(annos1[0].keys()) & set(smplx_shape.keys()))

    body_params1 = [{key: np.array(a1[key])
                    for key in intersect_key if len(a1[key]) > 0} for a1 in annos1]
    body_params2 = [{key: np.array(a2[key])
                    for key in intersect_key if len(a2[key]) > 0} for a2 in annos2]
    # for k,v in body_params1[0].items():
    #     print(f'{k}:{v.shape}')
    # import pdb;pdb.set_trace()



    for idx in range(n_temp):
        empty_img = np.zeros((img_height, img_width, 3),dtype=np.uint8)
        t = idx/(n_temp-1)
        frame_idx_tosave = idx


        # body_paramst = [{key: b1[key]*(1-t) + b2[key]*t for key in b1}
        #                 for b1,b2 in zip(body_params1,body_params2)]
        
        body_paramst = []
        for b1,b2 in zip(body_params1, body_params2):
            d = {}
            for key in b1:
                if key in direct_linear_keys:
                    d[key] = b1[key]*(1-t) + b2[key]*t
                else:
                    rotvec1 = torch.from_numpy(b1[key])
                    rotvec2 = torch.from_numpy(b2[key])
                    rotvec_mix = roma.rotvec_slerp(rotvec1.unsqueeze(0),
                                                    rotvec2.unsqueeze(0),
                                                    steps=torch.tensor([t]))[0][0]

                    d[key] = rotvec_mix.numpy()
                    # import pdb;pdb.set_trace()
                    # print(f'{key}:{d[key].shape}')
            body_paramst.append(d)

        metas = []
        for m1, m2 in zip(metas1, metas2):
            m = {}
            m['focal'] = [m1['focal'][0]*(1-t) + m2['focal'][0]*t, 
                        m1['focal'][1]*(1-t) + m2['focal'][1]*t]
            m['princpt'] = [m1['princpt'][0]*(1-t) + m2['princpt'][0]*t, 
                        m1['princpt'][1]*(1-t) + m2['princpt'][1]*t]
            metas.append(m)
        # save current smplx information
        all_pose_imgs = []
        for hid, smplx_param in enumerate(body_paramst):
            body_model_param_tensor = {key: torch.tensor(smplx_param[key].reshape(smplx_shape[key]), 
                                                        device=torch.device('cuda'), 
                                                        dtype=torch.float32)
                            for key in intersect_key if len(smplx_param[key]) > 0}
            output = body_model(**body_model_param_tensor, return_verts=True)

            joints = output['joints'].detach().cpu().numpy().squeeze()
            focal_length = metas[hid]['focal']
            principal_point = metas[hid]['princpt']
            fx=focal_length[0]
            fy=focal_length[1]
            cx=principal_point[0] 
            cy=principal_point[1]
            camera_pose = np.eye(4)
            proj_joints = project_vertices(joints, camera_pose, fx, fy, cx, cy)

            openpose_kps = proj_joints[reordered_smplx_idx][None,...]
            openpose_kps = np.concatenate([openpose_kps[:,:8], openpose_kps[:,9:]], axis=1)
            score = np.ones_like(openpose_kps[...,0])
            # import pdb;pdb.set_trace()
            pose = pose_drawer(openpose_kps,score, (img_height, img_width, 3))
            pose_img = draw_pose(pose)
            all_pose_imgs.append(pose_img)
            # import pdb;pdb.set_trace()
            
            if idx != 0 or idx != n_temp - 1:
                np.savez(f'{smplx_save_path}/frame_{idx:04d}_{hid+1}.npz', **smplx_param)

                json_object = json.dumps(metas[hid], indent=4)
                with open(os.path.join(smplx_meta_save_path, f'frame_{idx:04d}_{hid+1}.json'), "w") as outfile:
                    outfile.write(json_object)

        img,_ = render_annotation(body_paramst, metas, empty_img, body_model, use_vertex_color=True)
        # import pdb;pdb.set_trace()
        # cv2.imwrite('tmp.png', img)
        merged_pose_img = merge_pose_images(all_pose_imgs, img_height, img_width)
        cv2.imwrite(f"{smplx_pose_save_path}/frame_{frame_idx_tosave:04d}.png", cv2.cvtColor(merged_pose_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{smplx_cs_save_path}/frame_{frame_idx_tosave:04d}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # import pdb;pdb.set_trace()
if __name__ == "__main__":
    main()