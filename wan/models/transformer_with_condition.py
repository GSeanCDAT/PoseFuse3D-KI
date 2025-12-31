from diffusers import HunyuanVideoTransformer3DModel
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from wan.models.transformer_wan import WanTransformer3DModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanTransformer3DConditionModel(WanTransformer3DModel):
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        encoder_hidden_states_image_add: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        conditional_controls: Optional[torch.Tensor] = None,
        control_weight: float=1.,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image, encoder_hidden_states_image_add = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, encoder_hidden_states_image_add
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        cross_norm_dim = (3,)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for idx, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, encoder_hidden_states_image_add
                )

                # assert conditional_controls is not None
                if idx == 0 and conditional_controls is not None:
                    #b,t*h*w,c -> b*t,h,w,c
                    hidden_states = hidden_states.reshape(batch_size*post_patch_num_frames, post_patch_height, post_patch_width, -1)
                    mean_latents = torch.mean(hidden_states, dim=cross_norm_dim, keepdim=True)
                    std_latents = torch.std(hidden_states, dim=cross_norm_dim, keepdim=True)
                    scale = conditional_controls['scale']
                    conditional_controls = conditional_controls['output']
                    # print(conditional_controls.shape)
                    # NOTE:b,t*h*w,c -> b*t,h,w,c
                    conditional_controls = conditional_controls.reshape(batch_size*post_patch_num_frames, post_patch_height, post_patch_width, -1)
                    # print(conditional_controls.shape)
                    mean_control, std_control = torch.mean(conditional_controls, dim=cross_norm_dim, keepdim=True), torch.std(conditional_controls, dim=cross_norm_dim, keepdim=True)
                    conditional_controls = (conditional_controls - mean_control) * (std_latents / (std_control + 1e-5)) + mean_latents
                    # if hidden_states.shape[-2:]!=conditional_controls.shape[-2:]:
                    #     conditional_controls = F.adaptive_avg_pool2d(conditional_controls.permute(0,3,1,2), 
                    #                                                     hidden_states.shape[-2:]).permute(0,2,3,1)
                    #  0.2: This superparameter is used to adjust the control level: increasing this value will strengthen the control level.
                    # print([hidden_states.shape, conditional_controls.shape])
                    hidden_states = hidden_states + \
                        conditional_controls * scale * 0.2 * control_weight
                    
                    hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames*post_patch_height*post_patch_width, -1)

        else:
            for idx, block in enumerate(self.blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, encoder_hidden_states_image_add)
                # assert conditional_controls is not None
                if idx == 0 and conditional_controls is not None:
                    #b,t*h*w,c -> b*t,h,w,c
                    hidden_states = hidden_states.reshape(batch_size*post_patch_num_frames, post_patch_height, post_patch_width, -1)
                    mean_latents = torch.mean(hidden_states, dim=cross_norm_dim, keepdim=True)
                    std_latents = torch.std(hidden_states, dim=cross_norm_dim, keepdim=True)
                    scale = conditional_controls['scale']
                    conditional_controls = conditional_controls['output']
                    # print(conditional_controls.shape)
                    # NOTE:b,t*h*w,c -> b*t,h,w,c
                    conditional_controls = conditional_controls.reshape(batch_size*post_patch_num_frames, post_patch_height, post_patch_width, -1)
                    # print(conditional_controls.shape)
                    mean_control, std_control = torch.mean(conditional_controls, dim=cross_norm_dim, keepdim=True), torch.std(conditional_controls, dim=cross_norm_dim, keepdim=True)
                    conditional_controls = (conditional_controls - mean_control) * (std_latents / (std_control + 1e-5)) + mean_latents
                    # if hidden_states.shape[-2:]!=conditional_controls.shape[-2:]:
                    #     conditional_controls = F.adaptive_avg_pool2d(conditional_controls.permute(0,3,1,2), 
                    #                                                     hidden_states.shape[-2:]).permute(0,2,3,1)
                    #  0.2: This superparameter is used to adjust the control level: increasing this value will strengthen the control level.
                    # print([hidden_states.shape, conditional_controls.shape, scale.shape])
                    hidden_states = hidden_states + \
                        conditional_controls * scale * 0.2 * control_weight
                    
                    hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames*post_patch_height*post_patch_width, -1)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
