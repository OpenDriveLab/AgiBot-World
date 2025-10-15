"""PyTorch Action Expert(based on InternLM2) model, integrated to models from modeling_internlm2_go1.py."""

import math
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from go1.internvl.model.internlm2.modeling_internlm2 import InternLM2MLP, InternLM2RMSNorm

from .configuration_action_expert import ActionExpertConfig

flash_attn_func, flash_attn_varlen_func = None, None
pad_input, index_first_axis, unpad_input = None, None, None
try:
    from flash_attn import flash_attn_func as _flash_attn_func, flash_attn_varlen_func as _flash_attn_varlen_func
    from flash_attn.bert_padding import (
        index_first_axis as _index_first_axis,
        pad_input as _pad_input,
        unpad_input as _unpad_input,
    )

    flash_attn_func, flash_attn_varlen_func = _flash_attn_func, _flash_attn_varlen_func
    pad_input, index_first_axis, unpad_input = _pad_input, _index_first_axis, _unpad_input
    has_flash_attn = True
except:
    has_flash_attn = False
import logging

from go1.internvl.model.internlm2.modeling_internlm2 import (
    InternLM2DynamicNTKScalingRotaryEmbedding,
    InternLM2LinearScalingRotaryEmbedding,
    InternLM2RotaryEmbedding,
    _get_unpad_data,
    rearrange,
    repeat_kv,
    rotate_half,
)

# warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# Copied from transformers.model.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb_go1(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q: (B, H, horizon, head_dim)
        k: (B, H, vlm_seq_len, head_dim)
    """
    # make sure that the position_ids of action are consistent in q and k
    position_ids = position_ids + k.shape[2] - q.shape[2]
    cos_q = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin_q = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    position_ids_k = torch.arange(k.shape[2]).repeat(k.shape[0], 1)
    cos_k = cos[position_ids_k].unsqueeze(unsqueeze_dim)
    sin_k = sin[position_ids_k].unsqueeze(unsqueeze_dim)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed


class ActioinExpertAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.wqkv = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.bias,
        )

        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = InternLM2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.config.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "dynamic":
                self.rotary_emb = InternLM2DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "linear":
                self.rotary_emb = InternLM2LinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError("Currently we only support rotary embedding's type being 'dynamic' or 'linear'.")
        return self.rotary_emb

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        vlm_key_values: Tuple[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_k: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.wqkv(hidden_states)

        qkv_states = rearrange(
            qkv_states,
            "b q (h gs d) -> b q h gs d",
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )

        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]

        query_states = query_states.transpose(1, 2)  # （batch_size, head_num, vlm_seq_len, head_dim)
        key_states = key_states.transpose(1, 2)  # （batch_size, kv_head_num, vlm_seq_len, head_dim)
        value_states = value_states.transpose(1, 2)  # （batch_size, kv_head_num, vlm_seq_len, head_dim)

        key_states = torch.cat(
            [vlm_key_values[0], key_states], dim=2
        )  # （batch_size, kv_head_num, vlm_seq_len+horizon+1, head_dim)
        value_states = torch.cat(
            [vlm_key_values[1], value_states], dim=2
        )  # （batch_size, kv_head_num, vlm_seq_len+horizon+1, head_dim)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb_go1(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Modified from transformers.model.llama.modeling_llama.InternLM2FlashAttention2
class ActioinExpertAttentionFlashAttention2(ActioinExpertAttention):
    """
    InternLM2 flash attention module. This module inherits from `InternLM2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        vlm_key_values: Tuple[torch.Tensor],
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # InternLM2FlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.wqkv(hidden_states)

        qkv_states = rearrange(
            qkv_states,
            "b q (h gs d) -> b q h gs d",
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )

        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]

        query_states = query_states.transpose(1, 2)  # （batch_size, head_num, horizon+1, head_dim)
        key_states = key_states.transpose(1, 2)  # （batch_size, kv_head_num, horizon+1, head_dim)
        value_states = value_states.transpose(1, 2)  # （batch_size, kv_head_num, horizon+1, head_dim)

        key_states = torch.cat(
            [vlm_key_values[0], key_states], dim=2
        )  # （batch_size, kv_head_num, vlm_seq_len+horizon+1, head_dim)
        value_states = torch.cat(
            [vlm_key_values[1], value_states], dim=2
        )  # （batch_size, kv_head_num, vlm_seq_len+horizon+1, head_dim)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb_go1(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Contains at least one padding token in the sequence
        causal = self.is_causal and query_length != 1
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._unpad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output  # [bsz, tgt_len, num_head, head_dim]

    def _unpad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q.to(torch.int64),
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


ACTIONEXPERT_ATTENTION_CLASSES = {
    "eager": ActioinExpertAttention,
    "flash_attention_2": ActioinExpertAttentionFlashAttention2,
}


class ActionExpertDecoderLayer(nn.Module):
    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention = ACTIONEXPERT_ATTENTION_CLASSES[config.attn_implementation](config=config)

        self.feed_forward = InternLM2MLP(config)
        self.attention_norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        vlm_key_values: Tuple[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_k: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            vlm_key_values (`torch.FloatTensor`): vlm kv cache to the layer of shape `(batch, vlm_seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)

        # Self Attention with different kv and q length
        # kv is concatenation of vlm kv cache and state_action_traj embedding(done in attention layers)
        # q is the state_action_traj
        """
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            vlm_key_values=vlm_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        """
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            vlm_key_values=vlm_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_ids_k=position_ids_k,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class ActionExpertPretrainedModel(PreTrainedModel):
    config_class = ActionExpertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ActionExpertDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask_go1(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    state_length: int = 3,
    past_key_values_length: int = 4096,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask[:state_length, :state_length] = 0
    mask[state_length:, :] = 0
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _concat_mask_go1(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int):
    """
    Concates attention_mask using vlm mask `[bsz, vlm_len]` and get combined mask `[bsz, 1, tgt_len, src_len]`.
    """
    bsz, vlm_len = mask.size()
    src_len = vlm_len + tgt_len

    action_padding_mask = torch.ones((bsz, tgt_len), device=mask.device)
    concated_mask = torch.cat((mask, action_padding_mask), dim=-1)  # [bsz, src_len]
    expanded_mask = concated_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class ActionExpertModel(ActionExpertPretrainedModel):
    """Action Expert model based on InternLM2, the main difference is as follows:
    - hidden dim and intermediate dim shrinks to the half of original InternLM2-2B
    - text embedding layer replaced by action projection layers, converting action into hiddent states
    - initial action token initialized as noisy action tokens instead of input tokens
    - each docoder layer accepts kv states from corresponding LLM layer, and used as condition token for action
      generation
    - add new linear projector to squeeze kv states dim to align with action hidden states dim
    - causal mask for VLM tokens `I`, robotic states `q` and action tokens `a`
    """

    _auto_class = "AutoModel"

    def __init__(self, config: ActionExpertConfig):
        super().__init__(config)

        self.config = config
        self.rng_seeded = np.random.default_rng(seed=42)
        self.state_token_num = config.state_token_num
        if not has_flash_attn:
            self.config.attn_implementation = "eager"
            print("Warning: Flash attention is not available, using eager attention instead.")

        self.layers = nn.ModuleList([ActionExpertDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]

        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask_go1(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                state_length=self.state_token_num,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:  # [bsz, vlm_seq_len]
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _concat_mask_go1(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        # action related inputs
        state_action_traj: torch.FloatTensor,
        # vlm related inputs
        attention_mask: torch.Tensor = None,
        vlm_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # output related args
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = state_action_traj.shape[:2]
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = state_action_traj.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        dynamic_vlm_token_length = attention_mask.shape[-1]

        position_ids_k = None

        if self.config.attn_implementation == "flash_attention_2":
            attention_mask = None
        else:  # generate attention mask in shape [bsz, 1, tgt_len, src_len] using default attention layer
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, dynamic_vlm_token_length), dtype=torch.bool, device=state_action_traj.device
                )
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                input_shape=(batch_size, seq_length),
                inputs_embeds=state_action_traj,
                past_key_values_length=dynamic_vlm_token_length,
            )

        # embed positions
        hidden_states = state_action_traj

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    vlm_key_values[idx],
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )

            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    vlm_key_values=vlm_key_values[idx],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_ids_k=position_ids_k,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
