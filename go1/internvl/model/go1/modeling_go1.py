import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from timm.models.vision_transformer import Mlp
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from go1.internvl.conversation import get_conv_template
from go1.internvl.model.internlm2.modeling_internlm2 import InternLM2RMSNorm
from go1.internvl.model.internvl_chat.modeling_intern_vit import InternVisionModel
from go1.internvl.model.internvl_chat.modeling_internvl_chat import version_cmp

from .configuration_go1 import GO1ModelConfig
from .modeling_action_expert import ActionExpertModel
from .modeling_internlm2_go1 import InternLM2ForCausalLMGO1
from .modeling_latent_action_expert import LatentPlannerModel

logger = logging.get_logger(__name__)


@dataclass
class ActionModelOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    labels: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    action_loss: Optional[torch.FloatTensor] = None
    action_logits: Optional[torch.FloatTensor] = None
    action_gts: Optional[torch.FloatTensor] = None


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer of RDT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = InternLM2RMSNorm(hidden_size)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x


class GO1Model(PreTrainedModel):
    config_class = GO1ModelConfig
    main_input_name = "pixel_values"
    base_model_prefix = "language_model"
    _no_split_modules = [
        "InternVisionModel",
        "InternLM2DecoderLayerGO1",
        "ActionExpertDecoderLayer",
    ]
    _supports_flash_attn_2 = True

    def __init__(
        self,
        config: GO1ModelConfig,
        vision_model=None,
        language_model=None,
        action_model=None,
        torch_dtype=torch.bfloat16,
    ):
        super().__init__(config)
        assert version_cmp(transformers.__version__, "4.37.0", "ge")
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]
        self.torch_dtype = torch_dtype

        self.max_new_tokens = 1024
        self.img_context_token_id = getattr(config, "img_context_token_id", 92546)
        self.pad_token_id = config.pad_token_id

        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"ps_version: {self.ps_version}")
        if vision_model is not None:
            self.vision_model = vision_model
            if self.vision_model.dtype != self.torch_dtype:
                self.vision_model.to(self.torch_dtype)
        else:
            self.vision_model = InternVisionModel(config.vision_config).to(self.torch_dtype)
        if language_model is not None:
            self.language_model = language_model
            if language_model.dtype != self.torch_dtype:
                self.language_model.to(self.torch_dtype)
        else:
            if "InternLM2ForCausalLM" in config.llm_config.architectures[0]:
                self.language_model = InternLM2ForCausalLMGO1(config.llm_config).to(self.torch_dtype)
            else:
                raise NotImplementedError(f"{config.llm_config.architectures[0]} is not implemented.")

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        llm_head_dim = llm_hidden_size // config.llm_config.num_attention_heads

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, dtype=self.torch_dtype),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size, dtype=self.torch_dtype),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size, dtype=self.torch_dtype),
        )

        self.conv_template = get_conv_template(self.template)
        if hasattr(config, "system_message"):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

        # Action Expert related initialization
        action_config = config.action_config
        if action_model is not None:
            self.action_model = action_model
            if self.action_model.dtype != self.torch_dtype:
                self.action_model.to(self.torch_dtype)
        else:
            self.action_model = ActionExpertModel(action_config).to(self.torch_dtype)
        action_hidden_size = action_config.hidden_size
        self.action_dim = action_config.action_dim
        action_head_dim = action_hidden_size // action_config.num_attention_heads

        # latent action expert related initialization
        latent_planner_config = config.latent_planner_config
        if config.latent_planning:
            self.enable_lam = True
            self.latent_planner = LatentPlannerModel(
                action_head_dim,
                config,
                latent_planner_config,
                self.torch_dtype,
            )
            latent_action_head_dim = self.latent_planner.latent_action_head_dim
        else:
            self.enable_lam = False

        self.k_proj_layers = nn.ModuleList(
            [
                nn.Linear(
                    llm_head_dim,  # 128 to 64
                    latent_action_head_dim if config.latent_planning else action_head_dim,
                    dtype=self.torch_dtype,
                )
                for _ in range(config.llm_config.num_hidden_layers)
            ]
        )
        self.v_proj_layers = nn.ModuleList(
            [
                nn.Linear(
                    llm_head_dim,
                    latent_action_head_dim if config.latent_planning else action_head_dim,
                    dtype=self.torch_dtype,
                )
                for _ in range(config.llm_config.num_hidden_layers)
            ]
        )
        self.time_embedder = TimestepEmbedder(action_hidden_size, dtype=self.torch_dtype)
        self.freq_embedder = TimestepEmbedder(action_hidden_size, dtype=self.torch_dtype)

        self.state_adaptor = nn.Sequential(
            nn.Linear(
                (action_config.state_dim),
                action_config.hidden_size,
                dtype=self.torch_dtype,
            ),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_config.hidden_size, action_config.hidden_size, dtype=self.torch_dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_config.hidden_size, action_config.hidden_size, dtype=self.torch_dtype),
        )

        self.action_adaptor = nn.Sequential(
            nn.Linear(
                action_config.action_dim,
                action_config.hidden_size,
                dtype=self.torch_dtype,
            ),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_config.hidden_size, action_config.hidden_size, dtype=self.torch_dtype),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_config.hidden_size, action_config.hidden_size, dtype=self.torch_dtype),
        )

        self.final_layer = FinalLayer(action_hidden_size, self.action_dim).to(self.torch_dtype)
        self.init_linear_weights()

        # Noise scheduluar related initialization
        self.action_chunk_size = config.action_chunk_size
        noise_scheduler_config = config.noise_scheduler_config
        self.num_train_timesteps = noise_scheduler_config["num_train_timesteps"]
        self.num_inference_timesteps = noise_scheduler_config["num_inference_timesteps"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config["num_train_timesteps"],
            beta_schedule=noise_scheduler_config["beta_schedule"],
            prediction_type=noise_scheduler_config["prediction_type"],
            clip_sample=noise_scheduler_config["clip_sample"],
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config["num_train_timesteps"],
            beta_schedule=noise_scheduler_config["beta_schedule"],
            prediction_type=noise_scheduler_config["prediction_type"],
        )

    def init_linear_weights(self) -> None:
        # Initialize the kv cache projector
        for k_linear, v_linear in zip(self.k_proj_layers, self.v_proj_layers):
            nn.init.xavier_uniform_(k_linear.weight)
            nn.init.xavier_uniform_(v_linear.weight)
            if k_linear.bias is not None:
                nn.init.constant_(k_linear.bias, 0)
            if v_linear.bias is not None:
                nn.init.constant_(v_linear.bias, 0)

        # Initialize timestep and control freq embedding MLP
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)
        # Initialize the final layer: zero-out the final linear layer
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        if int(h * scale_factor) / scale_factor != h or int(w * scale_factor) / scale_factor != w:
            w_ = int(int(w * scale_factor) / scale_factor)
            h_ = int(int(h * scale_factor) / scale_factor)
            x = F.interpolate(x.permute(0, 3, 1, 2), size=(w_, h_), mode="bilinear", align_corners=False)
            x = x.permute(0, 2, 3, 1).contiguous()
            n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def calc_action_diffusion_loss(
        self,
        action_logits: torch.FloatTensor,
        action_gts: Optional[torch.Tensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        pred_type: str = "sample",
    ) -> torch.Tensor:
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = action_gts
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(action_logits, target)
        return loss

    def condition_sample(
        self,
        state: torch.Tensor,
        vlm_key_values_downsample: torch.FloatTensor,
        attention_mask: torch.Tensor,
        ctrl_freqs: int = 30,
    ):
        state_traj = self.state_adaptor(state)  # (B, 1, C)
        device, dtype = state_traj.device, state_traj.dtype
        # Sample noise that we'll add to the actions
        noisy_action = torch.randn(
            size=(state_traj.shape[0], self.action_chunk_size, self.action_dim), device=device, dtype=dtype
        )
        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)

        for t in self.noise_scheduler_sample.timesteps:
            action_traj = self.action_adaptor(noisy_action)  # (B, H, C)
            timestep_tokens = self.time_embedder(
                t * torch.ones_like(ctrl_freqs, dtype=ctrl_freqs.dtype).to(device)
            )  # (B, 1, C)
            freq_tokens = self.freq_embedder(ctrl_freqs)  # (B, 1, C)
            state_action_trajs_w_tfps = torch.cat([timestep_tokens, freq_tokens, state_traj, action_traj], dim=1)

            # Predict the model output
            model_output = self.action_model(state_action_trajs_w_tfps, attention_mask, vlm_key_values_downsample)

            state_action_output_tokens = model_output[0]
            action_output_tokens = state_action_output_tokens[:, -self.action_chunk_size :, ...]
            action_output = self.final_layer(action_output_tokens)

            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(action_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(dtype)

        return noisy_action

    def common_process(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = True,
        labels: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        # Align with original InternVL implementation style
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(-1, C)

        input_ids = input_ids.reshape(-1)

        selected = input_ids == self.img_context_token_id
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            raise ValueError(
                f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, vit_embeds.shape={vit_embeds.shape}"
            )

        input_embeds = input_embeds.reshape(B, N, C)

        vlm_outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=return_dict,
            labels=labels,
        )
        return vlm_outputs

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        state: torch.Tensor = None,
        ctrl_freqs: int = None,
        action_gts: torch.Tensor = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, ActionModelOutputWithPast]:
        vlm_outputs = self.common_process(
            pixel_values,
            input_ids,
            attention_mask,
            position_ids,
            image_flags,
            return_dict,
            labels=labels,
        )
        # Project vlm kv_cache head dim into action expert head dim
        vlm_key_values = vlm_outputs.past_key_values  # kv_cache(multi_head) of each decoder layer

        vlm_key_values_downsample = []
        for vlm_key_value, k_proj, v_proj in zip(vlm_key_values, self.k_proj_layers, self.v_proj_layers):
            vlm_key_values_downsample.append((k_proj(vlm_key_value[0]), v_proj(vlm_key_value[1])))

        B = input_ids.shape[0]

        if self.enable_lam:
            (
                latent_vlm_key_values_downsample,
                outputs_latent,
            ) = self.latent_planner(
                vlm_key_values_downsample,
                attention_mask,
            )

        action_loss = torch.tensor(0.0, dtype=state.dtype, device=state.device)
        action_logits = None
        if self.training:
            # Sample noise that we'll add to the actions
            noise = torch.randn(action_gts.shape, dtype=action_gts.dtype, device=action_gts.device)
            timesteps = torch.randint(0, self.num_train_timesteps, (B, 1), device=action_gts.device).long()
            timestep_tokens = self.time_embedder(timesteps)  # (B, 1, C)
            freq_tokens = self.freq_embedder(ctrl_freqs)  # (B, 1, C)
            noisy_action = self.noise_scheduler.add_noise(action_gts, noise, timesteps)  # (B, H, C)

            state_trajs = self.state_adaptor(state)
            action_trajs = self.action_adaptor(noisy_action)
            state_action_trajs = torch.cat([state_trajs, action_trajs], dim=1)

            state_action_trajs_w_tfps = torch.cat(
                [timestep_tokens, freq_tokens, state_action_trajs], dim=1
            )  # (B, H+3, C)

            # Action expert as diffusion head
            if self.enable_lam:
                vlm_key_values_downsample = latent_vlm_key_values_downsample
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            B, self.latent_planner.latent_token_nums, dtype=torch.bool, device=attention_mask.device
                        ),
                    ),
                    dim=1,
                )

            outputs = self.action_model(state_action_trajs_w_tfps, attention_mask, vlm_key_values_downsample)
            state_action_output_tokens = outputs[0]
            action_output_tokens = state_action_output_tokens[:, -self.action_chunk_size :, ...]
            action_logits = self.final_layer(action_output_tokens)

            # Calculate action mse loss using output action chunk
            action_loss = self.calc_action_diffusion_loss(
                action_logits=action_logits,
                action_gts=action_gts,
                pred_type="sample",
            )
        else:
            # Action expert as diffusion head
            if self.enable_lam:
                vlm_key_values_downsample = latent_vlm_key_values_downsample
                attention_mask = torch.cat(
                    (attention_mask, torch.ones(B, 4, dtype=torch.bool, device=attention_mask.device)), dim=1
                )

            action_logits = self.condition_sample(
                state,
                vlm_key_values_downsample,
                attention_mask,
                ctrl_freqs,
            )

        loss = action_loss

        if not return_dict:
            return (loss, action_logits)

        return ActionModelOutputWithPast(
            loss=loss,
            action_loss=action_loss,
            action_logits=action_logits,
            action_gts=action_gts,
        )
