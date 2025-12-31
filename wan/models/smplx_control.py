import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import HunyuanVideoDownsampleCausal3D
from wan.models.casual_spatiotemporal_resnet import CasualSpatioTemporalResBlock

class MeshgridResNet(nn.Module):
    def __init__(
        self,
        embed_channels=32,
        groups = 4,
        eps = 1e-6,
    ):
        super().__init__()

        self.conv0 = nn.Conv2d(2, embed_channels, 1)
        self.norm1 = torch.nn.GroupNorm(num_groups=groups, 
                                        num_channels=embed_channels, 
                                        eps=eps, affine=True)
        self.conv1 = nn.Conv2d(embed_channels, 
                               embed_channels, 
                               kernel_size=3, 
                               stride=1, padding=1)
        self.norm2 = torch.nn.GroupNorm(num_groups=groups, 
                                        num_channels=embed_channels, 
                                        eps=eps, affine=True)
        self.conv2 = nn.Conv2d(embed_channels, embed_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = nn.SiLU()

    def forward(self, input_tensor):

        input_tensor = self.conv0(input_tensor)
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        output_tensor = input_tensor + hidden_states

        return output_tensor


class JointPoseEmbedding(nn.Module):

    def __init__(
        self,
        embed_channels,
        add_channel,
        eps = 1e-6,
        pad_mode = "replicate",
    ):
        super().__init__()
        self.pad_mode = pad_mode
        self.time_causal_padding = (2, 0)
        self.joint_pose_embed = nn.Sequential(
                nn.Conv1d(3, 64, 1),
                nn.GroupNorm(2, 64),
                nn.ReLU(),
                nn.Conv1d(64, 128, 1),
                nn.GroupNorm(2, 128),
                nn.ReLU(),
                )

        self.conv0 = nn.Conv1d(128+add_channel, embed_channels, 1)

        self.norm1 = torch.nn.GroupNorm(num_groups=8, 
                                        num_channels=embed_channels, 
                                        eps=eps, affine=True)
        self.conv1 = nn.Conv1d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.norm2 = torch.nn.GroupNorm(num_groups=8, 
                                        num_channels=embed_channels, 
                                        eps=eps, affine=True)
        self.conv2 = nn.Conv1d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.nonlinearity = nn.SiLU()


    def forward(self, input_tensor, 
                add_tensor, shape_indicator):

        b,t,n_human = shape_indicator
        input_tensor = self.joint_pose_embed(input_tensor)
        input_tensor = self.conv0(torch.cat([input_tensor, add_tensor], dim=1))

        hidden_states = rearrange(input_tensor, '(b t n) d x -> (b x n) d t', b=b, t=t, n=n_human)
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)
        hidden_states = self.conv1(hidden_states)


        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)
        hidden_states = self.conv2(hidden_states)

        hidden_states = rearrange(hidden_states, '(b x n) d t -> (b t n) d x', b=b, t=t, n=n_human)

        output_tensor = input_tensor + hidden_states

        return output_tensor

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        #apply feedforward layer to input tensor
        return self.net(x)

class CustomAttention(nn.Module):
    def __init__(self, 
                 indim_q, 
                 indim_k, 
                 indim_v, 
                 attn_dim,
                 num_heads=8, dropout=0.0):
        super().__init__()
        assert attn_dim % num_heads == 0 
        
        self.num_heads = num_heads
        self.head_dim= attn_dim // num_heads
        
        # linear projections for Q, K, V
        self.w_q = nn.Linear(indim_q, attn_dim, bias=False)
        self.w_k = nn.Linear(indim_k, attn_dim, bias=False)
        self.w_v = nn.Linear(indim_v, attn_dim, bias=False)
        
        # output projection
        self.proj = nn.Linear(attn_dim, attn_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):

        B, Lq, _ = Q.shape
        
        # project
        q = self.w_q(Q)  # (B, Lq, dim_q)
        k = self.w_k(K)  # (B, Lk, dim_k)
        v = self.w_v(V)  # (B, Lk, dim_v)
        
        # reshape to heads
        def reshape(x, head_dim):
            # from (B, L, H*head_dim) to (B, H, L, head_dim)
            return x.view(B, -1, self.num_heads, head_dim).transpose(1,2)
        
        qh = reshape(q, self.head_dim)
        kh = reshape(k, self.head_dim)
        vh = reshape(v, self.head_dim)
        
        # # scaled dot-product
        # scores = torch.matmul(qh, kh.transpose(-2, -1))  # (B, H, Lq, Lk)
        # scores = scores / (self.head_dim ** 0.5)
        
        # if mask is not None:
        #     # mask should be broadcastable to (B, H, Lq, Lk)
        #     scores = scores + mask.unsqueeze(1)
        
        # attn = F.softmax(scores, dim=-1)
        # attn = self.dropout(attn)
        
        # attention output
        # out_h = torch.matmul(attn, vh)  # (B, H, Lq, d_vh)
        out_h = F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        # merge heads
        out = out_h.transpose(1,2).contiguous().view(B, Lq, -1)  # (B, Lq, dim_v)
        
        # final projection
        out = self.proj(out)
        return out


class RegAttentionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reg_norm = nn.LayerNorm(dim)
        self.reg_attn = CustomAttention(dim, dim, dim, 
                                                dim)
        self.reg_ffn = FFN(dim, dim)

    def forward(self, hidden_state):
        norm_hidden_state = self.reg_norm(hidden_state)
        norm_hidden_state = self.reg_attn(norm_hidden_state, norm_hidden_state, norm_hidden_state)
        hidden_state = hidden_state + norm_hidden_state
        hidden_state = hidden_state + self.reg_ffn(hidden_state)
        return hidden_state

class SMPLX_Encoder(nn.Module):

    def __init__(self, 
                 match_dim=32, 
                 coor3d_embed_dim=64, 
                 joint_embed_dim=256,
                 attn_dim=256,
                 scale=4, 
                 time_embed_dim=256,
                 ):
        super(SMPLX_Encoder, self).__init__()

        self.meshgrid_embedding = MeshgridResNet(embed_channels=match_dim)
        self.coor2d_embedding = nn.Linear(2, match_dim)
        self.coor3d_embedding = nn.Sequential(
            nn.Linear(3, coor3d_embed_dim), 
            nn.LayerNorm(coor3d_embed_dim),
            nn.Linear(coor3d_embed_dim, coor3d_embed_dim*2),
            nn.GELU(),
            nn.Linear(coor3d_embed_dim*2, coor3d_embed_dim*2),
        )

        reduce_in_mul = 0

        self.joint_embed = JointPoseEmbedding(embed_channels=joint_embed_dim, 
                                            add_channel=coor3d_embed_dim*2)
        # joint
        self.map_joint = CustomAttention(match_dim, match_dim, joint_embed_dim, 
                                        attn_dim)
        self.joint_ffn = FFN(attn_dim, attn_dim)
        reduce_in_mul += 1

        # vertex
        self.map_point = CustomAttention(match_dim, match_dim, coor3d_embed_dim*2, 
                                        attn_dim)
        self.point_ffn = FFN(attn_dim, attn_dim)
        reduce_in_mul += 1

        self.scale = scale

        self.conv_reduce_dim = nn.Conv2d(attn_dim*reduce_in_mul, attn_dim, 1)

        # 
        strides = [(2,1,1), (2,2,2)]

        self.down_res = CasualSpatioTemporalResBlock(
                    in_channels=attn_dim,
                    out_channels=attn_dim,
                    temb_channels=time_embed_dim,
                    groups=16,
                    eps=1e-6,
                ) 
        self.down_sample = nn.ModuleList()
        for idx, stride in enumerate(strides):

            self.down_sample.append(
                HunyuanVideoDownsampleCausal3D(
                    attn_dim,
                    out_channels=attn_dim,
                    padding=0,
                    stride=stride
                )
            )

        self.conv_out = nn.Conv2d(in_channels=attn_dim,
                                out_channels=320,
                                kernel_size=1,
                                stride=1,
                                )


    def forward(self, 
                points, # b,t,n_human,10475,5
                joints, # b,t,n_human,144,5
                joint_poses, # b,t,n_human,55 ,3
                height,
                width,
                t_emb
                ):
        dtype = torch.bfloat16
        device = points.device

        # initialize info
        b,t,n_human = points.shape[:3]
        points = points.reshape(b*t*n_human, -1, 5)
        joints = joints.reshape(b*t*n_human, -1, 5)
        joint_poses = joint_poses.reshape(b*t*n_human, -1, 3)
        points3d, points2d = points[..., :-2], points[..., -2:].flip(dims=[-1])
        joints3d, joints2d = joints[..., :55, :-2], joints[..., :55, -2:].flip(dims=[-1])

        scaled_h, scaled_w = height//self.scale, width//self.scale


        meshgrid = torch.meshgrid(torch.linspace(0, height-1, scaled_h, dtype=dtype, device=device), 
                                  torch.linspace(0, width-1, scaled_w, dtype=dtype, device=device), 
                                  indexing='ij')
        meshgrid = torch.stack(meshgrid, 
                               dim=-1).unsqueeze(0).repeat(b*t*n_human, 1, 1, 1)
        def normalize_coor2d(t, h, w):
            return torch.stack([t[...,0]/(h-1)*2-1, t[...,1]/(w-1)*2-1], 
                               dim=-1)
        meshgrid = normalize_coor2d(meshgrid, height, width)
        points2d = normalize_coor2d(points2d, height, width)
        joints2d = normalize_coor2d(joints2d, height, width)

        # get 2d anchor features
        meshgrid_fea = self.meshgrid_embedding(meshgrid.permute(0,3,1,2)
                                               ).permute(0,2,3,1).reshape(b*t*n_human, 
                                                                          scaled_h*scaled_w, 
                                                                          -1)
        joints2d_fea = self.coor2d_embedding(joints2d)
        points2d_fea = self.coor2d_embedding(points2d)

        # get 3d features 
        joints3d_fea = self.coor3d_embedding(joints3d)
        joints_fea = self.joint_embed(joint_poses.permute(0,2,1), 
                                    joints3d_fea.permute(0,2,1), 
                                    [b, t, n_human]).permute(0,2,1)
        fea2 = self.map_joint(meshgrid_fea, joints2d_fea, joints_fea)
        fea2 = fea2 + self.joint_ffn(fea2)
        
        points3d_fea = self.coor3d_embedding(points3d)
        fea1 = self.map_point(meshgrid_fea, points2d_fea, points3d_fea)
        fea1 = fea1 + self.point_ffn(fea1)

        fea = []
        fea.append(fea1)
        fea.append(fea2)

        fea = torch.cat(fea, dim=-1)
        fea = rearrange(fea, '(b t n) (h w) d -> b d t n h w', 
                        b=b, t=t, n=n_human, h=scaled_h).sum(3)

        fea = rearrange(fea, 'b d t h w -> (b t) d h w')
        fea = self.conv_reduce_dim(fea)
        fea = rearrange(fea, '(b t) d h w -> b d t h w', b=b)

        fea = self.down_sample[0](fea)
        b, t = fea.shape[0], fea.shape[2]
        image_only_indicator = torch.zeros(b, t, 
                                            dtype=fea.dtype, 
                                            device=fea.device)
        
        fea = rearrange(fea, 'b d t h w -> (b t) d h w')
        # import pdb;pdb.set_trace()
        # print([fea.shape, t_emb.shape, t, image_only_indicator.shape])
        fea = self.down_res(fea, t_emb.repeat_interleave(t, dim=0),
                         image_only_indicator=image_only_indicator)
        fea = rearrange(fea, '(b t) d h w -> b d t h w', b=b)
        fea = self.down_sample[1](fea)
        fea = rearrange(fea, 'b d t h w -> (b t) d h w')


        fea = self.conv_out(fea)
        
        return fea



