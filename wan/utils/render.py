import os
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import cv2
import torch
import pyrender
import trimesh
# import pandas as pd
import json

import smplx
from wan.utils.human_models.human_models import smpl_x
import copy
import json


smpl_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 'body_pose': (-1, 69)}
smplx_shape = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
        'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3), 'expression': (-1, 10)}
smplx_shape_except_expression = {'betas': (-1, 10), 'transl': (-1, 3), 'global_orient': (-1, 3), 
        'body_pose': (-1, 21, 3), 'left_hand_pose': (-1, 15, 3), 'right_hand_pose': (-1, 15, 3), 
        'leye_pose': (-1, 3), 'reye_pose': (-1, 3), 'jaw_pose': (-1, 3)}
# smplx_shape = smplx_shape_except_expression
COLOR_PATH = './smplx_color.pt'

def render_pose(img, body_model_param, body_model, camera, return_mask=False, use_vertex_color=False):

    # the inverse is same
    pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    
    output = body_model(**body_model_param, return_verts=True)
    
    vertices = output['vertices'].detach().cpu().numpy().squeeze()
    faces = body_model.faces

    # render material
    base_color = (1.0, 193/255, 193/255, 1.0)
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0,
            alphaMode='OPAQUE',
            baseColorFactor=base_color)
    
    material_new = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1,
            roughnessFactor=0.4,
            alphaMode='OPAQUE',
            emissiveFactor=(0.2, 0.2, 0.2),
            baseColorFactor=(0.7, 0.7, 0.7, 1))  
    material = material_new
    
    # get body mesh
    body_trimesh = trimesh.Trimesh(vertices, faces, process=False)

    if use_vertex_color:
        vertex_colors = torch.load(COLOR_PATH) / 255.
        body_trimesh.visual.vertex_colors = vertex_colors

    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, )#material=material)

    # prepare camera and light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    cam_pose = pyrender2opencv @ np.eye(4)
    
    # build scene
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                    ambient_light=(0.3, 0.3, 0.3))
    scene.add(camera, pose=cam_pose)
    scene.add(light, pose=cam_pose)
    scene.add(body_mesh, 'mesh')

    # render scene
    # os.environ["PYOPENGL_PLATFORM"] = "osmesa" # include this line if use in vscode
    r = pyrender.OffscreenRenderer(viewport_width=img.shape[1],
                                    viewport_height=img.shape[0],
                                    point_size=1.0)
    
    color, depth = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    # alpha = 1.0  # set transparency in [0.0, 1.0]
    # color[:, :, -1] = color[:, :, -1] * alpha
    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    img = img / 255
    # output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * img)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    output_img = (color[:, :, :] * valid_mask + (1 - valid_mask) * img)

    # output_img = color

    img = (output_img * 255).astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if return_mask:
        return img, valid_mask, (color * 255).astype(np.uint8), depth

    return img

class CustomShaderCache():
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram("shaders/mesh.vert", "shaders/mesh.frag", defines=defines)
        return self.program

def render_pose_normal(img, body_model_param, body_model, camera, return_mask=False):

    # the inverse is same
    pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    
    output = body_model(**body_model_param, return_verts=True)
    
    vertices = output['vertices'].detach().cpu().numpy().squeeze()
    faces = body_model.faces
    
    # get body mesh
    body_trimesh = trimesh.Trimesh(vertices, faces, process=False)


    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, )#material=material)

    # prepare camera and light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    cam_pose = pyrender2opencv @ np.eye(4)
    
    # build scene
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                    ambient_light=(0.3, 0.3, 0.3))
    scene.add(camera, pose=cam_pose)
    scene.add(light, pose=cam_pose)
    scene.add(body_mesh, 'mesh')

    # render scene
    # os.environ["PYOPENGL_PLATFORM"] = "osmesa" # include this line if use in vscode
    r = pyrender.OffscreenRenderer(viewport_width=img.shape[1],
                                    viewport_height=img.shape[0],
                                    point_size=1.0)
    r._renderer._program_cache = CustomShaderCache()


    color, depth = r.render(scene)
    color = color.astype(np.float32) / 255.0
    # alpha = 1.0  # set transparency in [0.0, 1.0]
    # color[:, :, -1] = color[:, :, -1] * alpha
    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    img = img / 255
    # output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * img)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    output_img = (color[:, :, :] * valid_mask + (1 - valid_mask) * img)

    # output_img = color

    img = (output_img * 255).astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if return_mask:
        return img, valid_mask, (color * 255).astype(np.uint8), depth

    return img

def render_multi_pose(img,
                      body_model_params,
                      body_model,
                      cameras,
                      use_vertex_color=False,
                      render_normal=False):

    masks, colors = [], []

    # calculate distance based on transl
    dists, valid_idx = [], []
    for i, body_model_param in enumerate(body_model_params):
        dist = np.linalg.norm(body_model_param['transl'].detach().cpu()) * 2/ (cameras[i].fx + cameras[i].fy)
        if dist not in dists:
            valid_idx.append(i)
            dists.append(dist)

    # pdb.set_trace()

    # select by valid idx
    body_model_params = [body_model_params[i] for i in valid_idx]
    cameras = [cameras[i] for i in valid_idx]

    # sort by dist

    body_model_params = [x for _, x in sorted(zip(dists, body_model_params), reverse=True)]
    cameras = [x for _, x in sorted(zip(dists, cameras), reverse=True)]


    # render separate masks
    depths = []
    for i, body_model_param in enumerate(body_model_params):

        _, mask, color, depth = render_pose_normal(
            img=img,
            body_model_param=body_model_param,
            body_model=body_model,
            camera=cameras[i],
            return_mask=True,
        ) if render_normal else render_pose(
            img=img,
            body_model_param=body_model_param,
            body_model=body_model,
            camera=cameras[i],
            return_mask=True,
            use_vertex_color=use_vertex_color,
        )
        masks.append(mask)
        colors.append(color)
        depths.append(depth)
    # sum masks
    mask_sum = np.sum(masks, axis=0)
    mask_all = (mask_sum > 0)

    # pp_occ = 1 - np.sum(mask_all) / np.sum(mask_sum)
    # overlay colors to img
    for i, color in enumerate(colors):
        mask = masks[i]
        img = img * (1 - mask) + color * mask
        depth_mask = depths[i] > 0
        if i==0:
            depth = depths[0] * (1 - depth_mask) + depths[0] * depth_mask
        else:
            depth = depth * (1 - depth_mask) + depths[i] * depth_mask
    img = img.astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img, depth


def render_annotation(annos, metas, image, smplx_model, use_vertex_color=False, render_normal=False):

    body_model_params = []
    cameras = []
    # bbox_sizes = []
    # pdb.set_trace()
    for anno, meta in zip(annos,metas):

        # bbox_size = meta['bbox'][2] * meta['bbox'][3]
        focal_length = meta['focal']
        principal_point = meta['princpt']
        camera = pyrender.camera.IntrinsicsCamera(
                fx=focal_length[0], fy=focal_length[1],
                cx=principal_point[0], cy=principal_point[1],)

        # prepare body model params
        intersect_key = list(set(anno.keys()) & set(smplx_shape.keys()))
        body_model_param_tensor = {key: torch.tensor(
                np.array(anno[key]).reshape(smplx_shape[key]), device=torch.device('cuda'), dtype=torch.float32)
                        for key in intersect_key if len(anno[key]) > 0}
        
        cameras.append(camera)
        body_model_params.append(body_model_param_tensor)
        # bbox_sizes.append(bbox_size)

    # render pose
    rendered_image, depth = render_multi_pose(img=image, 
                    body_model_params=body_model_params, 
                    body_model=smplx_model.to(torch.device('cuda')),
                    cameras=cameras,
                    use_vertex_color=use_vertex_color,
                    render_normal=render_normal)

    def depth_to_image(d):
        m = d != 0
        d[~m] = np.nan
        depth_foreground = d[m]  ## value in range [0, 1]
        processed_depth = np.full((m.shape[0], m.shape[1]), 0, dtype=np.uint8)

        min_val, max_val = np.min(depth_foreground), np.max(depth_foreground)
        depth_normalized_foreground = 1 - (
            (depth_foreground - min_val) / ((max_val - min_val) * 255/254)
        )  ## for visualization, foreground is 1 (white), background is 0 (black)
        depth_normalized_foreground = (depth_normalized_foreground * 255.0).astype(
            np.uint8
        )

        processed_depth[m] = depth_normalized_foreground

        # if d.ndim==2:
        #     d = d[:, :, None]
        return processed_depth
    # NOTE: change it to rgb
    return rendered_image[..., ::-1], depth_to_image(depth)

def render_frame(framestamp, anno_ps, image_base_path, seq, smplx_model, args):
    annos = [p for p in anno_ps if framestamp in os.path.basename(p)]
    annos = [p for p in annos if 'person' not in os.path.basename(p)]

    body_model_params = []
    cameras = []
    bbox_sizes = []
    try:
        # image_path = os.path.join(seq, f'0{framestamp}.jpg').replace(args.data_path, args.image_path)
        image_path = os.path.join(image_base_path, f'0{framestamp}.jpg')
        # pdb.set_trace()
        image = cv2.imread(image_path)
    except:

        pass
    # pdb.set_trace()
    for anno_p in annos:

        anno = dict(np.load(anno_p, allow_pickle=True))

        meta = json.load(open(os.path.join(seq, 'meta', 
                                        os.path.basename(anno_p).replace('.npz', '.json')
                                        )))

        bbox_size = meta['bbox'][2] * meta['bbox'][3]
        focal_length = meta['focal']
        principal_point = meta['princpt']
        camera = pyrender.camera.IntrinsicsCamera(
                fx=focal_length[0], fy=focal_length[1],
                cx=principal_point[0], cy=principal_point[1],)

        # prepare body model params
        intersect_key = list(set(anno.keys()) & set(smplx_shape.keys()))
        body_model_param_tensor = {key: torch.tensor(
                np.array(anno[key]).reshape(smplx_shape[key]), device=args.device, dtype=torch.float32)
                        for key in intersect_key if len(anno[key]) > 0}
        
        cameras.append(camera)
        body_model_params.append(body_model_param_tensor)
        bbox_sizes.append(bbox_size)

    # render pose
    if args.render_biggest_person == 'True':
        bid = bbox_sizes.index(max(bbox_sizes))
        rendered_image = render_pose(img=image,
                        body_model_param=body_model_params[bid],
                        body_model=smplx_model,
                        camera=cameras[bid])
    else:
        rendered_image = render_multi_pose(img=image, 
                        body_model_params=body_model_params, 
                        body_model=smplx_model,
                        cameras=cameras)

    sp = seq.replace(f'{args.data_path}{os.path.sep}', '')
    save_path = os.path.join(args.data_path, 'output', sp)
    os.makedirs(save_path, exist_ok=True)

    save_name = os.path.join(save_path, framestamp+'.jpg')
    cv2.imwrite(save_name, rendered_image)


def call_frame_render(args):
    return render_frame(*args)
            

if __name__ == '__main__':
    smplx_model = copy.deepcopy(smpl_x.layer['neutral'])
    meta_path = '/mnt/sfs-common/zjguo/codebase/hcvfi/baselines/FCVG/data/new_pexels_ready/smplx_ann/video_3325978_scene-0_scene-0_stride1_downsample0_170/meta/frame_0149_1.json'
    annotated_smplx_path = '/mnt/sfs-common/zjguo/codebase/hcvfi/baselines/FCVG/data/new_pexels_ready/smplx_ann/video_3325978_scene-0_scene-0_stride1_downsample0_170/smplx/frame_0149_1.npz'
    empty_img = np.zeros((1080, 1920, 3),dtype=np.uint8)
    smplx_anns = [np.load(annotated_smplx_path)]
    with open(meta_path, 'r') as file:
        meta = json.load(file)
    
    # rendered cs image is bgr
    img = render_annotation(smplx_anns, [meta], empty_img, smplx_model, use_vertex_color=True)
    # import pdb;pdb.set_trace()
    # cv2.imwrite('tmp.png', img)
