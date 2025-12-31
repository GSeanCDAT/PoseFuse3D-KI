import numpy as np
import torch
from diffusers import WanImageToVideoPipeline

from PIL import Image

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL
import regex as re
import torch
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

from wan.models.transformer_with_condition import WanTransformer3DConditionModel
from wan.models.transformer_with_condition_FLF2V import WanTransformer3DConditionModel as WanTransformer3DConditionModel_FLF2V
from wan.models.condition_module import MultiCtrlModel

from wan.models.wrapper import Wrapped_Model

import copy

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

NEGATIVE_PROMPT_TEMPLATE = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, walking backwards"

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

class WanImageToVideoInterpolationPipeline(WanImageToVideoPipeline):

    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        transformer: Union[WanTransformer3DConditionModel, WanTransformer3DConditionModel_FLF2V],
        vae: AutoencoderKLWan,
        scheduler: Union[FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler],
        tokenizer: Optional[AutoTokenizer] = None,
        text_encoder: Optional[UMT5EncoderModel] = None,
        control_model: Optional[MultiCtrlModel] = None,
        wrapped_model: Optional[Wrapped_Model] = None,
        flf2v_style: bool = False,
    ):
        super().__init__(
                        tokenizer,
                        text_encoder,
                        image_encoder,
                        image_processor,
                        transformer,
                        vae,
                        scheduler)

        self.register_modules(
            control_model=control_model,
            wrapped_model=wrapped_model
        )
        self.flf2v_style = flf2v_style

    def encode_image(self, image: PipelineImageInput):
        image = self.image_processor(images=image, return_tensors="pt").to(self.device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    def prepare_latents(
        self,
        image1: PipelineImageInput,
        image2: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image1 = image1.unsqueeze(2)

        video_condition1 = torch.cat(
            [image1, image1.new_zeros(image1.shape[0], image1.shape[1], num_frames - 1, height, width)], dim=2
        )
        video_condition1 = video_condition1.to(device=device, dtype=dtype)

        image2 = image2.unsqueeze(2)
        video_condition2 = torch.cat(
            [image2, image2.new_zeros(image2.shape[0], image2.shape[1], num_frames - 1, height, width)], dim=2
        )
        video_condition2 = video_condition2.to(device=device, dtype=dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        if isinstance(generator, list):
            latent_condition1 = [
                retrieve_latents(self.vae.encode(video_condition1), sample_mode="argmax") for _ in generator
            ]
            latent_condition1 = torch.cat(latent_condition1)
            latent_condition2 = [
                retrieve_latents(self.vae.encode(video_condition2), sample_mode="argmax") for _ in generator
            ]
            latent_condition2 = torch.cat(latent_condition2)
        else:
            latent_condition1 = retrieve_latents(self.vae.encode(video_condition1), sample_mode="argmax")
            latent_condition1 = latent_condition1.repeat(batch_size, 1, 1, 1, 1)
            latent_condition2 = retrieve_latents(self.vae.encode(video_condition2), sample_mode="argmax")
            latent_condition2 = latent_condition2.repeat(batch_size, 1, 1, 1, 1)

        latent_condition1 = (latent_condition1 - latents_mean) * latents_std
        latent_condition2 = (latent_condition2 - latents_mean) * latents_std

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
        mask_lat_size[:, :, list(range(1, num_frames))] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition1.device)

        return latents, torch.concat([mask_lat_size, latent_condition1], dim=1), torch.concat([mask_lat_size, latent_condition2], dim=1)

    def prepare_latents(
        self,
        image1: PipelineImageInput,
        image2: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image1 = image1.unsqueeze(2)
        image2 = image2.unsqueeze(2)

        video_condition1 = torch.cat(
            [image1, image1.new_zeros(image1.shape[0], image1.shape[1], num_frames - 2, height, width), image2], dim=2
        )
        video_condition1 = video_condition1.to(device=device, dtype=dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        if isinstance(generator, list):
            latent_condition1 = [
                retrieve_latents(self.vae.encode(video_condition1), sample_mode="argmax") for _ in generator
            ]
            latent_condition1 = torch.cat(latent_condition1)
        else:
            latent_condition1 = retrieve_latents(self.vae.encode(video_condition1), sample_mode="argmax")
            latent_condition1 = latent_condition1.repeat(batch_size, 1, 1, 1, 1)

        latent_condition1 = (latent_condition1 - latents_mean) * latents_std

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
        
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        if self.flf2v_style:
            mask_lat_size[:, :, list(range(1, num_frames-1))] = 0
            mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        else:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
            mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:-4, :], copy.deepcopy(first_frame_mask)], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition1.device)

        return latents, torch.concat([mask_lat_size, latent_condition1], dim=1), None

    @torch.no_grad()
    def single_step(self, 
                    latents, 
                    timestep, 
                    image_latent,  
                    control_model_condition1, 
                    control_model_condition2,
                    control_model_condition1_add,
                    control_model_condition2_add,
                    prompt_embeds, 
                    attention_kwargs,
                    avg_weight,
                    control_weight=1.0,
                    image1_embeddings=None, 
                    image2_embeddings=None, 
                    depth=None, 
                    normal=None, 
                    points=None, joints=None, joint_poses=None, 
                    cam_embed=None,
    ):  
        # print(timestep.shape)
        latents1 = latents
        latent_model_input1 = torch.cat([latents1] * 2) if self.do_classifier_free_guidance else latents1

        # image_latent = torch.cat([image1_latents[:, :, :-1], 
        #                           image2_latents[:, :, :1]], dim=2)

        image_latent = torch.cat([image_latent] * 2) if self.do_classifier_free_guidance else image_latent

        image1_embeddings = torch.cat([image1_embeddings] * 2) if self.do_classifier_free_guidance else image1_embeddings
        image2_embeddings = torch.cat([image2_embeddings] * 2) if self.do_classifier_free_guidance else image2_embeddings

        # Concatenate image_latents over channels dimention
        latent_model_input1 = torch.cat([latent_model_input1, image_latent], dim=1)
        if self.wrapped_model is not None:
            transformer_input_kwargs1 = dict(
                hidden_states=latent_model_input1,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image1_embeddings,
                encoder_hidden_states_image_add=image2_embeddings,
                attention_kwargs=attention_kwargs,
                return_dict=False,
                control_weight=control_weight
            )
            noise_pred1 = self.wrapped_model(control_model_condition1, control_model_condition1_add, timestep,
                                             transformer_input_kwargs1,
                                             cfg=self.do_classifier_free_guidance, 
                                             depth=depth, 
                                             normal=normal, 
                                             points=points, joints=joints, joint_poses=joint_poses, cam_embed=cam_embed)
            
        else:
            if self.control_model is not None:
                # import pdb;pdb.set_trace()
                control_model_output1 = self.control_model(control_model_condition1, control_model_condition1_add, timestep, depth=depth, normal=normal, 
                                                           points=points, joints=joints, joint_poses=joint_poses, cam_embed=cam_embed)
                if self.do_classifier_free_guidance:
                    N = control_model_output1['output'].shape[0]*latents1.shape[2]
                    control_model_output1['scale'] = torch.tensor(control_model_output1['scale']).to(latent_model_input1).repeat(N)[:, None, None, None]
                    control_model_output1['scale'][:N // 2] *= 0
            else:
                control_model_output1=None

            noise_pred1 = self.transformer(
                hidden_states=latent_model_input1,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image1_embeddings,
                encoder_hidden_states_image_add=image2_embeddings,
                attention_kwargs=attention_kwargs,
                return_dict=False,
                conditional_controls=control_model_output1,
                control_weight=control_weight
            )[0]

        # perform guidance TODO: verify the input latent for cfg
        if self.do_classifier_free_guidance:
            noise_pred_uncond1, noise_pred_cond1 = noise_pred1.chunk(2)
            noise_pred1 = noise_pred_uncond1 + self.guidance_scale * (noise_pred_cond1 - noise_pred_uncond1)

        return noise_pred1

    @torch.no_grad()
    def __call__(
        self,
        image1: PipelineImageInput,
        image2: PipelineImageInput,
        control_model_condition: Optional[torch.FloatTensor] = None,
        control_model_condition_add: Optional[torch.FloatTensor] = None,
        prompt: Union[str, List[str]] = '',
        negative_prompt: Union[str, List[str]] = NEGATIVE_PROMPT_TEMPLATE,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        control_weight: float = 1.0, 
        depth: Optional[torch.FloatTensor] = None,
        normal: Optional[torch.FloatTensor] = None,
        points: Optional[torch.FloatTensor] = None,
        joints: Optional[torch.FloatTensor] = None,
        joint_poses: Optional[torch.FloatTensor] = None,
        cam_embed: Optional[torch.FloatTensor] = None,
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        if prompt_embeds is not None:
            prompt = None
        if negative_prompt_embeds is not None:
            negative_prompt = None
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            image1,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Encode image embedding
        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
            
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds]) if self.do_classifier_free_guidance else torch.cat([prompt_embeds, prompt_embeds])

        image_embeds1 = self.encode_image(image1)
        image_embeds1 = image_embeds1.repeat(batch_size, 1, 1)
        image_embeds1 = image_embeds1.to(transformer_dtype)

        image_embeds2 = self.encode_image(image2)
        image_embeds2 = image_embeds2.repeat(batch_size, 1, 1)
        image_embeds2 = image_embeds2.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim
        image1 = self.video_processor.preprocess(image1, height=height, width=width).to(device, dtype=torch.float32)
        image2 = self.video_processor.preprocess(image2, height=height, width=width).to(device, dtype=torch.float32)
        
        latents, condition1, condition2 = self.prepare_latents(
            image1,
            image2,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # process controlmodel conditions
        if control_model_condition is not None:
            if not isinstance(control_model_condition, torch.Tensor):
                control_model_condition = self.image_processor.preprocess(control_model_condition, height=height, width=width)
                control_model_condition = (control_model_condition + 1.0) / 2
            if control_model_condition.ndim == 4:
                control_model_condition = control_model_condition.unsqueeze(0)
            if self.do_classifier_free_guidance:
                control_model_condition = torch.cat([control_model_condition] * 2) 
            control_model_condition = control_model_condition.to(device, latents.dtype)
        #prepare controlmodel condition 2 (3d cond)
        def prepare_cond_tensor(t):
            if t is not None:
                if not isinstance(t, torch.Tensor):
                    t = self.image_processor.preprocess(t, height=height, width=width)
                    t = (t + 1.0) / 2
                if t.ndim == 4:
                    t = t.unsqueeze(0)
                if self.do_classifier_free_guidance:
                    t = torch.cat([t] * 2) 
                t = t.to(device, latents.dtype)
            return t
        
        control_model_condition_add = prepare_cond_tensor(control_model_condition_add)
        depth = prepare_cond_tensor(depth)
        normal = prepare_cond_tensor(normal)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # NOTE: temporal dimension is the third dimension 
        w = torch.linspace(1, 0, num_latent_frames).unsqueeze(0).unsqueeze(0).to(device, latents.dtype)
        w = w.repeat(batch_size*num_videos_per_prompt, 1, 1)
        w = _append_dims(w, latents.ndim)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                timestep = t.expand(latents.shape[0])
                if control_model_condition is not None and control_model_condition_add is not None:
                    control_model_condition1 = control_model_condition
                    control_model_condition2 = torch.flip(control_model_condition1, (1,))

                    control_model_condition1_add = control_model_condition_add
                    control_model_condition2_add = torch.flip(control_model_condition1_add, (1,))
                else:
                    control_model_condition1 = None
                    control_model_condition2 = None
                    control_model_condition1_add = None
                    control_model_condition2_add = None
                noise_pred = self.single_step(latents, 
                                                timestep, 
                                                condition1, 
                                                control_model_condition1, 
                                                control_model_condition2,
                                                control_model_condition1_add,
                                                control_model_condition2_add,
                                                # DiT input kwargs
                                                prompt_embeds, 
                                                attention_kwargs,
                                                w, control_weight,
                                                image1_embeddings=image_embeds1, 
                                                image2_embeddings=image_embeds2, 
                                                depth=depth,
                                                normal=normal, 
                                                points=prepare_cond_tensor(points), 
                                                joints=prepare_cond_tensor(joints), 
                                                joint_poses=prepare_cond_tensor(joint_poses),
                                                cam_embed=prepare_cond_tensor(cam_embed),
                                                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)