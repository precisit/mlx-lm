# Copyright © 2026 Apple Inc.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_ssm_mask
from .cache import ArraysCache, CacheList, KVCache
from .gated_delta import gated_delta_update
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False

    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    attention_bias: bool = False
    max_position_embeddings: int = 262144
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000.0
    rope_parameters: Optional[Dict[str, Any]] = None

    full_attention_interval: int = 4
    layer_types: List[str] = field(default_factory=list)

    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: Optional[str] = "sigmoid"
    mamba_ssm_dtype: str = "float32"

    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4

    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True

    hc_count: int = 4
    hc_lowrank: int = 320

    ple_layer_ids: List[int] = field(default_factory=list)
    ple_embed_dim: Optional[int] = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    split_ngram_parts: int = 128
    ngram_shard_rows: int = 0
    seed: int = 1234
    eos_token_id: Union[int, List[int]] = 248044

    def __post_init__(self):
        self.output_gate_type = self.output_gate_type or self.hidden_act
        if self.rope_parameters:
            self.partial_rotary_factor = self.rope_parameters.get(
                "partial_rotary_factor", self.partial_rotary_factor
            )
            self.rope_theta = self.rope_parameters.get(
                "rope_theta", self.rope_theta
            )

        if not self.layer_types:
            self.layer_types = [
                "linear_attention"
                if (i + 1) % self.full_attention_interval
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must have one entry per decoder layer.")
        if set(self.layer_types) - {"linear_attention", "full_attention"}:
            raise ValueError("Only linear_attention and full_attention are supported.")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_key_value_heads must divide num_attention_heads."
            )
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("linear_num_key_heads must divide linear_num_value_heads.")
        indexer_values = (
            self.indexer_n_heads,
            self.indexer_kv_heads,
            self.indexer_head_dim,
            self.indexer_budget,
            self.indexer_compress_ratio,
        )
        if any(value <= 0 for value in indexer_values):
            raise ValueError("QSA indexer dimensions and budgets must be positive.")
        if self.indexer_kv_heads != 1:
            raise ValueError("Qwen4-Exp QSA requires indexer_kv_heads=1.")
        if self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError(
                "indexer_budget must be divisible by indexer_compress_ratio."
            )
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if rotary_dim > self.indexer_head_dim:
            raise ValueError("The attention RoPE dimensions must fit the QSA head.")
        if self.output_gate_type not in {"sigmoid", "silu"}:
            raise ValueError("output_gate_type must be sigmoid or silu.")
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts].")
        if self.hc_count <= 1:
            raise ValueError("Qwen4-Exp requires hc_count greater than one.")
        if self.ple_embed_dim is None:
            self.ple_embed_dim = self.hidden_size
        ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if ngram_heads <= 0:
            raise ValueError("The n-gram head count must be positive.")
        if self.ple_layer_ids and self.ple_embed_dim % ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by the n-gram head count."
            )
        for layer_id in self.ple_layer_ids:
            if not 1 <= layer_id <= self.num_hidden_layers:
                raise ValueError("PLE layer IDs are one-indexed decoder layer IDs.")
            if self.layer_types[layer_id - 1] != "linear_attention":
                raise ValueError("PLE is supported only on linear-attention layers.")


class Qwen4ExpRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, group_size: Optional[int] = None):
        super().__init__()
        if group_size is not None and dim % group_size:
            raise ValueError(
                "The RMSNorm dimension must be divisible by its group size."
            )
        self.weight = mx.zeros(dim)
        self.eps = eps
        self.group_size = group_size

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
        x = x.astype(mx.float32)
        x = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-2], -1)
        return (x * (1.0 + self.weight.astype(mx.float32))).astype(dtype)


class Qwen4ExpRMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float, activation: str):
        super().__init__()
        self.weight = mx.ones(dim)
        self.eps = eps
        self.activation = activation

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        x = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        x = (x.astype(dtype) * self.weight).astype(mx.float32)
        if self.activation == "sigmoid":
            gate = mx.sigmoid(gate.astype(mx.float32))
        else:
            gate = nn.silu(gate.astype(mx.float32))
        return (x * gate).astype(dtype)


class Qwen4ExpGatedDeltaNet(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = args.linear_conv_kernel_dim

        self.in_proj_qkv = nn.Linear(args.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(args.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
        )
        self.dt_bias = mx.ones(self.num_v_heads)
        self.A_log = mx.log(
            mx.random.uniform(low=0.01, high=16, shape=(self.num_v_heads,))
        )
        self.norm = Qwen4ExpRMSNormGated(
            self.head_v_dim, args.rms_norm_eps, args.output_gate_type
        )
        self.out_proj = nn.Linear(self.value_dim, args.hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[ArraysCache] = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(
            batch, length, self.num_v_heads, self.head_v_dim
        )
        beta = self.in_proj_b(x)
        decay = self.in_proj_a(x)

        state = cache[0] if cache is not None else None
        if state is None:
            state = mx.zeros(
                (batch, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype
            )
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([state, qkv], axis=1)
        if cache is not None:
            cache[0] = mx.contiguous(
                conv_input[:, -(self.conv_kernel_size - 1) :, :]
            )
        qkv = nn.silu(self.conv1d(conv_input))

        query, key, value = [
            value.reshape(batch, length, heads, dim)
            for value, heads, dim in zip(
                mx.split(qkv, [self.key_dim, 2 * self.key_dim], axis=-1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        query_dtype = query.dtype
        key_dtype = key.dtype
        query = query.astype(mx.float32)
        key = key.astype(mx.float32)
        query = query * mx.rsqrt(
            mx.sum(mx.square(query), axis=-1, keepdims=True) + 1e-6
        )
        key = key * mx.rsqrt(
            mx.sum(mx.square(key), axis=-1, keepdims=True) + 1e-6
        )
        query = (query * self.head_k_dim**-0.5).astype(query_dtype)
        key = key.astype(key_dtype)
        recurrent_state = cache[1] if cache is not None else None
        output, recurrent_state = gated_delta_update(
            query,
            key,
            value,
            decay,
            beta,
            self.A_log,
            self.dt_bias,
            recurrent_state,
            mask,
            use_kernel=not self.training and self.head_k_dim % 32 == 0,
        )
        if cache is not None:
            cache[1] = recurrent_state
            cache.advance(length)

        output = self.norm(output, z).reshape(batch, length, self.value_dim)
        return self.out_proj(output)


class Qwen4ExpQSAIndexer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.indexer_n_heads
        self.head_dim = args.indexer_head_dim
        self.token_budget = args.indexer_budget
        self.compress_ratio = args.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.rope_theta = args.rope_theta
        self.rope = initialize_rope(
            self.rotary_dim,
            base=args.rope_theta,
            traditional=False,
            max_position_embeddings=args.max_position_embeddings,
        )
        self.index_qk_proj = nn.Linear(
            args.hidden_size,
            (args.indexer_n_heads + args.indexer_kv_heads)
            * args.indexer_head_dim,
            bias=False,
        )
        self.q_layernorm = Qwen4ExpRMSNorm(
            self.head_dim, args.rms_norm_eps
        )
        self.k_layernorm = Qwen4ExpRMSNorm(
            self.head_dim, args.rms_norm_eps
        )

    def _rope_blocks(self, x: mx.array, positions: mx.array) -> mx.array:
        batch, num_blocks, dim = x.shape
        if num_blocks == 0:
            return x
        x = x.reshape(batch * num_blocks, 1, 1, dim)
        offsets = mx.broadcast_to(positions[None], (batch, num_blocks)).reshape(-1)
        x = mx.fast.rope(
            x,
            self.rotary_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=offsets,
        )
        return x.reshape(batch, num_blocks, dim)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: Optional[ArraysCache],
    ) -> tuple[mx.array, mx.array]:
        batch, length, _ = x.shape
        qk = self.index_qk_proj(x)
        query, raw_keys = mx.split(
            qk, [self.num_heads * self.head_dim], axis=-1
        )
        query = query.reshape(batch, length, self.num_heads, self.head_dim)
        query = self.q_layernorm(query).transpose(0, 2, 1, 3)
        query = self.rope(query, offset=offset).transpose(0, 2, 1, 3)

        if cache is not None:
            if cache[0] is not None:
                raw_keys = mx.concatenate([cache[0], raw_keys], axis=1)
            cache[0] = raw_keys

        total_length = raw_keys.shape[1]
        ratio = self.compress_ratio
        num_blocks = total_length // ratio
        query_positions = offset + mx.arange(length)
        complete_blocks = (query_positions + 1) // ratio

        if num_blocks:
            block_keys = raw_keys[:, : num_blocks * ratio]
            block_keys = block_keys.reshape(
                batch, num_blocks, ratio, self.head_dim
            )
            block_keys = block_keys.astype(mx.float32).mean(axis=2).astype(
                raw_keys.dtype
            )
            block_keys = self.k_layernorm(block_keys)
            block_starts = mx.arange(num_blocks) * ratio
            block_keys = self._rope_blocks(block_keys, block_starts)
            scores = mx.einsum(
                "blhd,bnd->blnh",
                query.astype(mx.float32),
                block_keys.astype(mx.float32),
            )
            scores = mx.maximum(scores, 0).sum(axis=-1) / self.head_dim**0.5
            block_ids = mx.arange(num_blocks)
            allowed = block_ids[None, None, :] < complete_blocks[None, :, None]
            scores = mx.where(allowed, scores, mx.finfo(mx.float32).min)
            indices, valid = _qsa_select_tokens(
                scores,
                query_positions,
                ratio,
                self.block_topk,
            )
        else:
            scores = mx.zeros((batch, length, 0), dtype=mx.float32)
            indices, valid = _qsa_select_tokens(
                scores,
                query_positions,
                ratio,
                self.block_topk,
            )
        return indices, valid


def _qsa_select_tokens(
    scores: mx.array,
    query_positions: mx.array,
    compress_ratio: int,
    block_topk: int,
) -> tuple[mx.array, mx.array]:
    batch, length, num_blocks = scores.shape
    complete_blocks = (query_positions + 1) // compress_ratio
    if num_blocks:
        block_ids = mx.arange(num_blocks)
        allowed = block_ids[None, None, :] < complete_blocks[None, :, None]
        scores = mx.where(allowed, scores, mx.finfo(scores.dtype).min)
        k = min(block_topk, num_blocks)
        selected_blocks = mx.argpartition(scores, kth=-k, axis=-1)[..., -k:]
        selected_valid = selected_blocks < complete_blocks[None, :, None]
        selected_tokens = (
            selected_blocks[..., None] * compress_ratio
            + mx.arange(compress_ratio)
        ).reshape(batch, length, -1)
        selected_valid = mx.broadcast_to(
            selected_valid[..., None],
            (*selected_valid.shape, compress_ratio),
        ).reshape(batch, length, -1)
    else:
        selected_tokens = mx.zeros((batch, length, 0), dtype=mx.int64)
        selected_valid = mx.zeros((batch, length, 0), dtype=mx.bool_)

    tail_width = compress_ratio - 1
    tail_start = complete_blocks * compress_ratio
    tail_tokens = tail_start[None, :, None] + mx.arange(tail_width)
    tail_tokens = mx.broadcast_to(tail_tokens, (batch, length, tail_width))
    tail_valid = tail_tokens <= query_positions[None, :, None]
    indices = mx.concatenate([selected_tokens, tail_tokens], axis=-1)
    valid = mx.concatenate([selected_valid, tail_valid], axis=-1)
    return mx.where(valid, indices, 0), valid


def _sparse_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    indices: mx.array,
    valid: mx.array,
    scale: float,
) -> mx.array:
    batch, query_heads, length, dim = queries.shape
    kv_heads = keys.shape[1]
    groups = query_heads // kv_heads
    width = indices.shape[-1]

    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)
    keys = mx.broadcast_to(
        keys[:, None], (batch, length, keys.shape[1], kv_heads, dim)
    )
    values = mx.broadcast_to(
        values[:, None], (batch, length, values.shape[1], kv_heads, dim)
    )
    gather_indices = mx.broadcast_to(
        indices[..., None, None], (batch, length, width, kv_heads, dim)
    )
    keys = mx.take_along_axis(keys, gather_indices, axis=2)
    values = mx.take_along_axis(values, gather_indices, axis=2)

    queries = queries.transpose(0, 2, 1, 3).reshape(
        batch, length, kv_heads, groups, dim
    )
    scores = mx.einsum(
        "blhgd,blmhd->blhgm",
        queries.astype(mx.float32),
        keys.astype(mx.float32),
    )
    scores = scores * scale
    scores = mx.where(
        valid[:, :, None, None, :], scores, mx.finfo(mx.float32).min
    )
    scores = mx.softmax(scores, axis=-1, precise=True)
    output = mx.einsum(
        "blhgm,blmhd->blhgd", scores.astype(values.dtype), values
    )
    return output.reshape(batch, length, query_heads * dim)


class Qwen4ExpSparseAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim**-0.5
        self.q_proj = nn.Linear(
            args.hidden_size,
            args.num_attention_heads * args.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * args.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * args.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * args.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.q_norm = Qwen4ExpRMSNorm(args.head_dim, args.rms_norm_eps)
        self.k_norm = Qwen4ExpRMSNorm(args.head_dim, args.rms_norm_eps)
        self.rope = initialize_rope(
            int(args.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=False,
            max_position_embeddings=args.max_position_embeddings,
        )
        self.indexer = Qwen4ExpQSAIndexer(args)

    def __call__(
        self, x: mx.array, cache: Optional[CacheList] = None
    ) -> mx.array:
        batch, length, _ = x.shape
        kv_cache = cache[0] if cache is not None else None
        indexer_cache = cache[1] if cache is not None else None
        if kv_cache is not None and not isinstance(kv_cache, KVCache):
            raise ValueError(
                "Qwen4-Exp QSA supports only an unquantized full KV cache."
            )
        offset = kv_cache.offset if kv_cache is not None else 0
        indices, valid = self.indexer(x, offset, indexer_cache)

        query, gate = mx.split(
            self.q_proj(x).reshape(
                batch, length, self.num_heads, 2 * self.head_dim
            ),
            2,
            axis=-1,
        )
        gate = gate.reshape(batch, length, -1)
        query = self.q_norm(query).transpose(0, 2, 1, 3)
        key = self.k_norm(
            self.k_proj(x).reshape(
                batch, length, self.num_kv_heads, self.head_dim
            )
        ).transpose(0, 2, 1, 3)
        value = self.v_proj(x).reshape(
            batch, length, self.num_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        query = self.rope(query, offset=offset)
        key = self.rope(key, offset=offset)
        if kv_cache is not None:
            key, value = kv_cache.update_and_fetch(key, value)

        output = _sparse_attention(
            query, key, value, indices, valid, self.scale
        )
        return self.o_proj(output * mx.sigmoid(gate))


class Qwen4ExpMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen4ExpSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_expert = Qwen4ExpMLP(
            args.hidden_size, args.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        probabilities = mx.softmax(self.gate(x), axis=-1, precise=True)
        indices = mx.argpartition(
            probabilities, kth=-self.top_k, axis=-1
        )[..., -self.top_k :]
        scores = mx.take_along_axis(probabilities, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        routed = self.switch_mlp(x, indices)
        routed = (routed * scores[..., None]).sum(axis=-2)
        shared = self.shared_expert(x)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * shared
        return routed + shared


class Qwen4ExpGatedResidual(nn.Module):
    def __init__(self, args: ModelArgs, use_injection: bool = True):
        super().__init__()
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        hc_size = args.hc_count * args.hidden_size
        self.hc_norm = Qwen4ExpRMSNorm(
            hc_size, args.rms_norm_eps, group_size=args.hidden_size
        )
        self.input_mix_weight_down = nn.Linear(
            hc_size, args.hc_lowrank, bias=False
        )
        self.input_mix_weight_up = nn.Linear(
            args.hc_lowrank, hc_size, bias=False
        )
        if use_injection:
            self.block_inject_weight = nn.Linear(
                hc_size, args.hc_count, bias=False
            )
        else:
            self.block_inject_weight = None

    def __call__(self, x: mx.array):
        x_norm = self.hc_norm(x)
        mix = nn.silu(self.input_mix_weight_down(x_norm) / self.hc_count)
        mix = mx.sigmoid(self.input_mix_weight_up(mix))
        mix = mix.reshape(
            *mix.shape[:-1], self.hc_count, self.hidden_size
        )
        streams = x_norm.reshape(
            *x_norm.shape[:-1], self.hc_count, self.hidden_size
        )
        mixed = (mix * streams).mean(axis=-2)
        if self.block_inject_weight is None:
            return mixed
        injection = 2 * mx.sigmoid(
            self.block_inject_weight(x_norm) / self.hc_count
        )
        return mixed, x, injection


class Qwen4ExpShardedEmbedding(nn.Module):
    def __init__(self, num_shards: int, rows: int, dims: int):
        super().__init__()
        self.num_shards = num_shards
        self.shard_rows = rows
        self.dims = dims
        self._head_shards = None
        if rows <= 0:
            raise ValueError("The package must provide an n-gram shard row count.")
        for index in range(num_shards):
            setattr(self, f"shard_{index}", nn.Embedding(rows, dims))

    def set_head_ranges(self, offsets: mx.array, sizes: mx.array):
        offsets = offsets.tolist()
        sizes = sizes.tolist()
        head_shards = []
        for offset, size in zip(offsets, sizes):
            end = offset + size
            shards = []
            for shard_index in range(self.num_shards):
                shard_start = shard_index * self.shard_rows
                shard_end = shard_start + self.shard_rows
                if offset < shard_end and end > shard_start:
                    shards.append(shard_index)
            if not shards:
                raise ValueError("An n-gram head is outside the sharded table.")
            head_shards.append(tuple(shards))
        self._head_shards = tuple(head_shards)

    def __call__(
        self,
        indices: mx.array,
        offsets: mx.array,
        sizes: mx.array,
    ) -> mx.array:
        if self._head_shards is None:
            self.set_head_ranges(offsets, sizes)
        if indices.shape[-1] != len(self._head_shards):
            raise ValueError("The n-gram ID and head counts do not match.")

        head_outputs = []
        for head_index, shard_indices in enumerate(self._head_shards):
            head_ids = indices[..., head_index]
            output = None
            for shard_index in shard_indices:
                start = shard_index * self.shard_rows
                shard = getattr(self, f"shard_{shard_index}")
                local = mx.clip(head_ids - start, 0, self.shard_rows - 1)
                values = shard(local)
                selected = (head_ids >= start) & (
                    head_ids < start + self.shard_rows
                )
                values = mx.where(selected[..., None], values, 0)
                output = values if output is None else output + values
            head_outputs.append(output)
        return mx.stack(head_outputs, axis=-2)


class Qwen4ExpNGramEmbedding(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ngram_size = args.ngram_size
        self.context_len = args.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = self.context_len * self.heads_per_ngram
        self.eos_token_id = (
            args.eos_token_id[0]
            if isinstance(args.eos_token_id, list)
            else args.eos_token_id
        )
        head_dim = args.ple_embed_dim // self.ngram_heads
        self.layer_multipliers = mx.zeros(args.ngram_size, dtype=mx.int64)
        self.ngram_heads_vocab_sizes = mx.zeros(
            self.ngram_heads, dtype=mx.int64
        )
        self.ngram_heads_offsets = mx.zeros(
            self.ngram_heads, dtype=mx.int64
        )
        self.ngram_embedding = Qwen4ExpShardedEmbedding(
            args.split_ngram_parts, args.ngram_shard_rows, head_dim
        )

    def _shift_right_ignore_eos(
        self, token_ids: mx.array, shift: int
    ) -> mx.array:
        if shift == 0:
            return token_ids
        batch, length = token_ids.shape
        positions = mx.arange(length, dtype=mx.int64)
        eos_positions = mx.where(
            token_ids == self.eos_token_id, positions[None], -1
        )
        previous_eos = mx.cummax(eos_positions, axis=1)
        previous_eos = mx.concatenate(
            [mx.full((batch, 1), -1, dtype=mx.int64), previous_eos[:, :-1]],
            axis=1,
        )
        segment_start = previous_eos + 1
        source_positions = positions - shift
        gather_positions = mx.broadcast_to(
            mx.maximum(source_positions, 0)[None], (batch, length)
        )
        shifted = mx.take_along_axis(token_ids, gather_positions, axis=1)
        valid = (positions[None] - segment_start >= shift) & (
            source_positions[None] >= 0
        )
        return mx.where(valid, shifted, self.eos_token_id)

    def __call__(
        self, input_ids: mx.array, cache: Optional[ArraysCache]
    ) -> mx.array:
        input_ids = input_ids.astype(mx.int64)
        batch, length = input_ids.shape
        previous = cache[3] if cache is not None else None
        if previous is None:
            previous = mx.full(
                (batch, self.context_len), self.eos_token_id, dtype=mx.int64
            )
        history = mx.concatenate([previous, input_ids], axis=1)
        if cache is not None:
            cache[3] = mx.contiguous(history[:, -self.context_len :])

        shifted = [
            self._shift_right_ignore_eos(history, shift)
            for shift in range(self.ngram_size)
        ]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed = mx.bitwise_xor(
                    mixed, shifted[position] * self.layer_multipliers[position]
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = mx.remainder(mixed[..., None], sizes[None, None])
            blocks.append(ids + offsets[None, None])
        ngram_ids = mx.concatenate(blocks, axis=-1)[:, -length:]
        return self.ngram_embedding(
            ngram_ids,
            self.ngram_heads_offsets,
            self.ngram_heads_vocab_sizes,
        ).reshape(batch, length, -1)


class Qwen4ExpPLE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_size = args.hidden_size * args.hc_count
        self.ple_embedding = Qwen4ExpNGramEmbedding(args)
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_size, bias=False)
        self.value_proj = nn.Linear(
            args.ple_embed_dim, args.hidden_size, bias=False
        )
        self.norm_key = Qwen4ExpRMSNorm(
            hc_size, args.rms_norm_eps, group_size=args.hidden_size
        )
        self.norm_query = Qwen4ExpRMSNorm(
            hc_size, args.rms_norm_eps, group_size=args.hidden_size
        )
        self.norm_conv = Qwen4ExpRMSNorm(
            hc_size, args.rms_norm_eps, group_size=args.hidden_size
        )
        self.conv_dilation = args.ngram_size
        self.conv_kernel_size = args.ple_conv_kernel_size
        self.conv_state_size = (self.conv_kernel_size - 1) * self.conv_dilation
        self.conv1d = nn.Conv1d(
            hc_size,
            hc_size,
            self.conv_kernel_size,
            dilation=self.conv_dilation,
            groups=hc_size,
            bias=False,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        mask: Optional[mx.array],
        cache: Optional[ArraysCache],
    ) -> mx.array:
        embeddings = self.ple_embedding(input_ids, cache)
        key = self.norm_key(self.key_proj(embeddings)).reshape(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        value = self.value_proj(embeddings)
        query = self.norm_query(hidden_states).reshape(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        gate = (key * query).sum(axis=-1, keepdims=True) / self.hidden_size**0.5
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = mx.sigmoid(gate) * value[..., None, :]
        gated = gated.reshape(*hidden_states.shape)
        gated_norm = self.norm_conv(gated)
        if mask is not None:
            gated = mx.where(mask[..., None], gated, 0)
            gated_norm = mx.where(mask[..., None], gated_norm, 0)

        state = cache[2] if cache is not None else None
        if state is None:
            state = mx.zeros(
                (hidden_states.shape[0], self.conv_state_size, gated.shape[-1]),
                dtype=gated.dtype,
            )
        conv_input = mx.concatenate([state, gated_norm], axis=1)
        if cache is not None:
            cache[2] = mx.contiguous(conv_input[:, -self.conv_state_size :, :])
        return gated + nn.silu(self.conv1d(conv_input))


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        if self.is_linear:
            self.linear_attn = Qwen4ExpGatedDeltaNet(args)
        else:
            self.self_attn = Qwen4ExpSparseAttention(args)
        self.mlp = Qwen4ExpSparseMoeBlock(args)
        self.ple = Qwen4ExpPLE(args) if layer_idx + 1 in args.ple_layer_ids else None
        self.attn_hyper_connection = Qwen4ExpGatedResidual(args)
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(args)

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Union[ArraysCache, CacheList]],
    ) -> mx.array:
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(
                hidden_states, input_ids, mask, cache
            )

        mixed, residual, injection = self.attn_hyper_connection(hidden_states)
        if self.is_linear:
            output = self.linear_attn(mixed, mask, cache)
        else:
            output = self.self_attn(mixed, cache)
        hidden_states = residual + (
            output[..., None, :] * injection[..., None]
        ).reshape(*residual.shape)

        mixed, residual, injection = self.mlp_hyper_connection(hidden_states)
        output = self.mlp(mixed)
        return residual + (
            output[..., None, :] * injection[..., None]
        ).reshape(*residual.shape)


class Qwen4ExpTextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(
            args, use_injection=False
        )

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            if input_ids is None and self.args.ple_layer_ids:
                raise ValueError("PLE requires input IDs with input embeddings.")
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(input_ids)
        if cache is None:
            cache = [None] * len(self.layers)

        linear_index = next(
            (i for i, layer in enumerate(self.layers) if layer.is_linear), None
        )
        mask = (
            create_ssm_mask(hidden_states, cache[linear_index])
            if linear_index is not None
            else None
        )
        hidden_states = mx.tile(hidden_states, (1, 1, self.args.hc_count))
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(
                hidden_states, input_ids, mask, layer_cache
            )
        return self.hyper_connection_mixer(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpTextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        output = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(output)
        return self.lm_head(output)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(4 if layer.ple is not None else 2))
            else:
                caches.append(CacheList(KVCache(), ArraysCache(1)))
        return caches

    def sanitize(self, weights):
        weights = {
            key: value for key, value in weights.items() if not key.startswith("mtp.")
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_id in self.args.ple_layer_ids:
            prefix = f"model.layers.{layer_id - 1}.ple.ple_embedding"
            sizes_key = f"{prefix}.ngram_heads_vocab_sizes"
            offsets_key = f"{prefix}.ngram_heads_offsets"
            if sizes_key not in weights or offsets_key not in weights:
                raise ValueError("The PLE n-gram hash buffers are required.")
            required_rows = int(
                mx.max(weights[sizes_key] + weights[offsets_key]).item()
            )
            shard_keys = [
                f"{prefix}.ngram_embedding.shard_{index}.weight"
                for index in range(self.args.split_ngram_parts)
            ]
            if any(key not in weights for key in shard_keys):
                raise ValueError("All configured n-gram shards are required.")
            row_counts = [weights[key].shape[0] for key in shard_keys]
            if any(rows != row_counts[0] for rows in row_counts):
                raise ValueError("All n-gram shards must have the same row count.")
            shard_rows = sum(row_counts)
            if shard_rows < required_rows:
                raise ValueError(
                    "The n-gram shards provide "
                    f"{shard_rows} rows but the hash buffers require "
                    f"{required_rows}."
                )
            self.layers[layer_id - 1].ple.ple_embedding.ngram_embedding.set_head_ranges(
                weights[offsets_key], weights[sizes_key]
            )

        for layer_index in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_index}.mlp"
            gate_up = f"{prefix}.experts.gate_up_proj"
            if gate_up not in weights:
                continue
            for suffix in ("", ".scales", ".biases"):
                source = gate_up + suffix
                if source not in weights:
                    continue
                value = weights.pop(source)
                middle = value.shape[-2] // 2
                target_suffix = ".weight" if not suffix else suffix
                weights[
                    f"{prefix}.switch_mlp.gate_proj{target_suffix}"
                ] = value[..., :middle, :]
                weights[
                    f"{prefix}.switch_mlp.up_proj{target_suffix}"
                ] = value[..., middle:, :]

            down = f"{prefix}.experts.down_proj"
            for suffix in ("", ".scales", ".biases"):
                source = down + suffix
                if source in weights:
                    target_suffix = ".weight" if not suffix else suffix
                    weights[
                        f"{prefix}.switch_mlp.down_proj{target_suffix}"
                    ] = weights.pop(source)

        for key, value in list(weights.items()):
            if key.endswith("conv1d.weight") and value.ndim == 3:
                if value.shape[1] == 1:
                    weights[key] = value.moveaxis(2, 1)
        return weights

    @property
    def cast_predicate(self):
        return lambda path: not path.endswith("A_log")
