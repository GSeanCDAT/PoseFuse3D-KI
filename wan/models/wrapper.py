import ftfy
import html
import re

import torch
import torch.nn as nn

from transformers import AutoTokenizer, UMT5EncoderModel
from typing import List, Optional, Union

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text

class TextWrapper(nn.Module):

    def __init__(self,
                text_encoder: UMT5EncoderModel,
                tokenizer: AutoTokenizer,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

class Wrapped_Model(nn.Module):
    def __init__(self, 
                 transformer, 
                 control_model):
        super().__init__()
        self.transformer = transformer
        self.control_model = control_model

    def forward(self, all_conds_2d,
                all_conds_3d, timesteps,
                transformer_input_kwargs, cfg=False, 
                depth=None, normal=None,
                points=None, joints=None, joint_poses=None, 
                cam_embed=None):
        if self.control_model is not None:
            conditional_controls = self.control_model(all_conds_2d, 
                                                all_conds_3d, 
                                                timesteps,
                                                depth=depth, 
                                                normal=normal,
                                                points=points, joints=joints, joint_poses=joint_poses, cam_embed=cam_embed)
            if cfg:
                N = conditional_controls['output'].shape[0] * \
                    transformer_input_kwargs['hidden_states'].shape[2]
                conditional_controls['scale'] = \
                    torch.tensor(conditional_controls['scale']).to(transformer_input_kwargs['hidden_states']).repeat(N)[:, None, None, None]
                conditional_controls['scale'][:N // 2] *= 0
        else:
            conditional_controls = None

        transformer_input_kwargs["conditional_controls"] = conditional_controls
        model_pred = self.transformer(**transformer_input_kwargs)[0]
        return model_pred