from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from functools import partial

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import Downsample2D, ResnetBlock2D
from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import HunyuanVideoDownsampleCausal3D
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoPatchEmbed

from wan.models.customized_transformer import CustomizedTransformerSpatioTemporalModel
from wan.models.casual_spatiotemporal_resnet import CasualSpatioTemporalResBlock
from wan.models.smplx_control import SMPLX_Encoder


class MultiCtrlModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        add_temporal_layer=True,
        time_embed_dim = 256,
        out_channels = 320,
        num_attention_heads = 5, 
        num_transformer_layers = 2, 
        cross_attention_dim = 320,
        win_len=[5,5],
        patchembed_dim=5120,
        depth_cond=False,
        normal_cond=False,
        smplx_cond=False,
    ):
        super().__init__()

        # the default 2d controlnext as 2d control module
        self.control_model_2d = SimplifiedControlNeXt(add_temporal_layer=add_temporal_layer)
        self.control_model_3d = SimplifiedControlNeXt(add_temporal_layer=add_temporal_layer, 
                                                    depth_cond=False,
                                                    normal_cond=False)
        if depth_cond:
            self.depth_control = SimplifiedControlNeXt(add_temporal_layer=add_temporal_layer, out_channels=[128, 128],
                                                    input_channel=1)
        self.depth_cond = depth_cond
        self.normal_cond = normal_cond
        if normal_cond:
            self.normal_control = SimplifiedControlNeXt(add_temporal_layer=add_temporal_layer, out_channels=[128, 128],)
        
        self.condition_fusion = CustomizedTransformerSpatioTemporalModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=num_transformer_layers,
                    cross_attention_dim=cross_attention_dim,
                    win_len=win_len,
                )


        self.smplx_cond = smplx_cond
        if self.smplx_cond:
            self.smplx_encoder = SMPLX_Encoder()

        self.time_proj = Timesteps(128, True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(128, time_embed_dim)

        self.downsample_module = HunyuanVideoPatchEmbed(
                    patch_size=(1,2,2),
                    in_chans=out_channels,
                    embed_dim=patchembed_dim,
                )

        self.scale = 1. 

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def enable_gradient_checkpointing(self) -> None:
        """
        Activates gradient checkpointing for the current model (may be referred to as *activation checkpointing* or
        *checkpoint activations* in other frameworks).
        """
        if not self._supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        self.apply(partial(self._set_gradient_checkpointing, value=True))
            

    # Copied from diffusers.models.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def forward(
        self,
        sample_2d: torch.Tensor,
        sample_3d: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        ref_latent: Optional[torch.Tensor] = None,
        output_dtype: Optional[torch.dtype] = None,
        depth: Optional[torch.Tensor] = None,
        normal: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        joints: Optional[torch.Tensor] = None,
        joint_poses: Optional[torch.Tensor] = None,
        cam_embed: Optional[torch.Tensor] = None,
    ):
        
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample_2d.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample_2d.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample_2d.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample_2d.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample_2d.dtype)
        # print([t_emb.shape, self.time_embedding.linear_1.weight.shape, self.time_embedding.linear_1.bias.shape])
        # import pdb;pdb.set_trace()
        emb_batch = self.time_embedding(t_emb)

        # Flatten the batch and frames dimensions
        # sample_(2/3)d: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample_2d = sample_2d.flatten(0, 1)
        sample_3d = sample_3d.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb_batch #.repeat_interleave(num_frames, dim=0)
        image_only_indicator = torch.zeros(batch_size, num_frames, 
                                            dtype=sample_2d.dtype, 
                                            device=sample_2d.device)
        encoded_2d = self.control_model_2d(sample_2d, emb, image_only_indicator)
        # encoded_3d = self.control_model_3d(sample_3d, emb, image_only_indicator, depth=depth.flatten(0, 1), 
        #                                    normal=normal.flatten(0, 1))
        encoded_3d = [self.control_model_3d(sample_3d, emb, image_only_indicator)]
        # assert self.smplx_cond
        if self.smplx_cond:
            encoded_3d.append(self.smplx_encoder(points, joints, joint_poses,
                                            sample_2d.shape[-2], sample_2d.shape[-1], emb)
                                            )
        
        # assert self.depth_cond and self.normal_cond
        # print([depth.shape, normal.shape])
        if self.depth_cond:
            # import pdb;pdb.set_trace()
            encoded_3d.append(self.depth_control(depth.flatten(0, 1), emb, image_only_indicator))
        if self.normal_cond:
            # import pdb;pdb.set_trace()
            encoded_3d.append(self.normal_control(normal.flatten(0, 1), emb, image_only_indicator))

        # print([a.shape for a in encoded_3d])
        encoded_3d = torch.stack(encoded_3d, dim=0).sum(0)

        num_frames = encoded_2d.shape[0]//batch_size
        image_only_indicator = torch.zeros(batch_size, num_frames, 
                                            dtype=sample_2d.dtype, 
                                            device=sample_2d.device)
        sample = self.condition_fusion(encoded_2d, encoded_3d, image_only_indicator, ref_img_latent=ref_latent)[0]

        _, channels, height, width = sample.shape
        sample = sample.reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        sample = self.downsample_module(sample)

        return {
            'output': sample.to(output_dtype) if output_dtype is not None else sample,
            'scale': self.scale,
        }


class SimplifiedControlNeXt(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        time_embed_dim = 256,
        in_channels = [128, 128],
        out_channels = [128, 256],
        groups = [4, 8],
        add_temporal_layer = False,
        depth_cond = False,
        normal_cond = False,
        input_channel = 3,
        out_dim = 320,
    ):
        super().__init__()

        self.add_temporal_layer = add_temporal_layer
        self.embedding = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(2, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(2, 128),
            nn.ReLU(),
            )

        self.depth_cond = depth_cond
        if depth_cond:
            self.depth_embedding = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(2, 64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.GroupNorm(2, 64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.GroupNorm(2, 128),
                nn.ReLU(),
                )
            
        self.normal_cond = normal_cond
        if normal_cond:
            self.normal_embedding = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(2, 64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.GroupNorm(2, 64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.GroupNorm(2, 128),
                nn.ReLU(),
                )

        self.down_res = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        for i in range(len(in_channels)):
            self.down_res.append(
                CasualSpatioTemporalResBlock(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    temb_channels=time_embed_dim,
                    groups=groups[i],
                    eps=1e-6,
                ) if self.add_temporal_layer else 
                ResnetBlock2D(
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    temb_channels=time_embed_dim,
                    groups=groups[i]
                )
            )
            self.down_sample.append(
                HunyuanVideoDownsampleCausal3D(
                    out_channels[i],
                    out_channels=out_channels[i],
                    padding=0,
                    stride=(2,2,2)
                )
            )
        
        self.mid_convs = nn.ModuleList()
        self.mid_convs.append(nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.GroupNorm(8, out_channels[-1]),
            nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.GroupNorm(8, out_channels[-1]),
        ))
        self.mid_convs.append(
            nn.Conv2d(
            in_channels=out_channels[-1],
            out_channels=out_dim,
            kernel_size=1,
            stride=1,
        ))

        self.scale = 1. 

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def forward(
        self,
        sample: torch.Tensor,
        emb: torch.Tensor,
        image_only_indicator: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        normal: Optional[torch.Tensor] = None,
    ):
        # sample: [batch * frames, channels, height, width]
        num_frames = image_only_indicator.shape[-1]
        # import pdb;pdb.set_trace()
        sample = self.embedding(sample)
        if self.depth_cond:
            assert False
            sample += self.depth_embedding(depth)
        if self.normal_cond:
            assert False
            sample += self.normal_embedding(normal)
        # sample = self.merge(sample)

        batch_frames, _, height, width = sample.shape
        batch_size = batch_frames // num_frames
        
        for idx, res in enumerate(self.down_res):
            sample = res(sample, emb.repeat_interleave(num_frames, dim=0),
                         image_only_indicator=image_only_indicator) if self.add_temporal_layer else res(sample, emb.repeat_interleave(num_frames, dim=0)) 
            # import pdb;pdb.set_trace()
            sample = sample.reshape(batch_size, num_frames, -1, height, width).permute(0, 2, 1, 3, 4)
            sample = self.down_sample[idx](sample)
            _, _, num_frames, height, width = sample.shape
            image_only_indicator = torch.zeros(batch_size, num_frames, 
                                                dtype=sample.dtype, 
                                                device=sample.device)
            batch_frames = batch_size*num_frames
            sample = sample.permute(0, 2, 1, 3, 4).reshape(batch_frames, -1, height, width)
        
        sample = self.mid_convs[0](sample) + sample
        sample = self.mid_convs[1](sample)
        # print(sample.shape)
        return sample