from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import torch
from torch import nn

# from diffusers.models.attention import TemporalBasicTransformerBlock

from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.resnet import AlphaBlender
from diffusers.models.embeddings import get_1d_rotary_pos_embed

from wan.models.customized_attention import BasicTransformerBlock, TemporalRopeBasicTransformerBlock

class CustomizedTransformerSpatioTemporalModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 5,
        attention_head_dim: int = 64,
        in_channels: int = 320,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
        temp_cross_attention_dim: Optional[int] = None, 
        win_len: List[int] = [5,5],
        theta: float = 10000.0,
    ):
        super().__init__()
        self.win_len = win_len
        self.theta = theta

        self.proj_multi_res = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim

        # 2. Define input layers
        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.norm_add = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.proj_in_add = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    use_window_cross_attn=True,
                    win_len=win_len
                )
                for d in range(num_layers)
            ]
        )

        self.transformer_blocks_add = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    double_self_attention=True,
                )
                for d in range(num_layers)
            ]
        )

        self.proj_out_adds = nn.ModuleList(
                [
                    nn.Linear(inner_dim, cross_attention_dim)
                    for d in range(num_layers)
                ]
            )
        
        
        self.norm_out_adds = nn.ModuleList(
                [
                    torch.nn.GroupNorm(num_groups=32, 
                                        num_channels=cross_attention_dim, 
                                        eps=1e-6)
                    for d in range(num_layers)
                ]
            )

        time_mix_inner_dim = inner_dim
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                TemporalRopeBasicTransformerBlock(
                    inner_dim,
                    time_mix_inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=temp_cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.use_temp_cross_attn = False if temp_cross_attention_dim is None else True
        self.time_mixer = AlphaBlender(alpha=0.5, merge_strategy="learned_with_images")
        

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        # TODO: should use out_channels for continuous projections
        self.proj_out = nn.Linear(inner_dim, in_channels)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_add: torch.Tensor,
        image_only_indicator: Optional[torch.Tensor] = None,
        ref_img_latent: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input hidden_states.
            num_frames (`int`):
                The number of frames to be processed per batch. This is used to reshape the hidden states.
            encoder_hidden_states (`(batch size, , channel, height, width)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            image_only_indicator (`torch.LongTensor` of shape `(batch size, num_frames)`, *optional*):
                A tensor indicating whether the input contains only images. 1 indicates that the input contains only
                images, 0 indicates that the input contains video frames.
        """
        # 1. Input
        batch_frames, _, height, width = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        batch_size = batch_frames // num_frames

        if self.use_temp_cross_attn:
            #TODO modifty this
            assert False
            time_context = encoder_hidden_states
            time_context_first_timestep = time_context[None, :].reshape(
                batch_size, num_frames, -1, time_context.shape[-1]
            )[:, 0]
            time_context = time_context_first_timestep[None, :].broadcast_to(
                height * width, batch_size, 1, time_context.shape[-1]
            )
            time_context = time_context.reshape(height * width * batch_size, 1, time_context.shape[-1])
        else:
            time_context = None

        # NOTE:process both hidden_states and hidden_states add
        residual = hidden_states
        residual_add = hidden_states_add

        hidden_states = self.norm(hidden_states)
        hidden_states_add = self.norm_add(hidden_states_add)

        inner_dim = hidden_states.shape[1]
        inner_dim_add = hidden_states_add.shape[1]

        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)
        hidden_states_add = hidden_states_add.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim_add)
        hidden_states_add = self.proj_in_add(hidden_states_add)


        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        
        temporal_rotary_emb = \
                get_1d_rotary_pos_embed(self.attention_head_dim, num_frames_emb, self.theta, use_real=True)
        # 2. Blocks
        for layer_idx, inner_blocks in enumerate(zip(self.transformer_blocks, 
                                         self.transformer_blocks_add, 
                                         self.temporal_transformer_blocks)):
            block, block_add, temporal_block = inner_blocks
            if self.training and self.gradient_checkpointing:
                hidden_states_add = torch.utils.checkpoint.checkpoint(
                    block_add,
                    hidden_states_add,
                    None,
                    None,
                    None,
                    use_reentrant=False,
                )
                # target size is [batch size*num_frames, H*W, D]
                cross_attn_states = self.proj_out_adds[layer_idx](hidden_states_add)
                cross_attn_states = cross_attn_states.reshape(batch_frames, height, 
                                                              width, inner_dim_add).permute(0, 3, 1, 2).contiguous()
                cross_attn_states = cross_attn_states + residual_add
                cross_attn_states = \
                    self.norm_out_adds[layer_idx](cross_attn_states).permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim_add)

                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    None,
                    cross_attn_states,
                    None,
                    image_size=[height, width],
                    use_reentrant=False,
                )

            else:
                hidden_states_add = block_add(
                    hidden_states_add,
                    encoder_hidden_states=None,
                )

                # target size is [batch size*num_frames, H*W, D]
                cross_attn_states = self.proj_out_adds[layer_idx](hidden_states_add)
                cross_attn_states = cross_attn_states.reshape(batch_frames, height, 
                                                              width, inner_dim_add).permute(0, 3, 1, 2).contiguous()
                cross_attn_states = cross_attn_states + residual_add
                cross_attn_states = \
                    self.norm_out_adds[layer_idx](cross_attn_states).permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim_add)
                
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=cross_attn_states,
                    image_size=[height, width],
                )

            hidden_states_mix = hidden_states

            hidden_states_mix = temporal_block(
                hidden_states_mix,
                num_frames=num_frames,
                encoder_hidden_states=time_context,
                image_rotary_emb=temporal_rotary_emb
            )
            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )
        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()


        residual = self.proj_multi_res(
                    torch.cat([residual, residual_add], dim=1)
                    )

        output = hidden_states + residual

        return (output,)