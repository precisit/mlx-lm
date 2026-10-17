# Qwen4-Exp text runtime package contract

This runtime consumes text-only MLX packages emitted by the sibling
`precisit-gemma4` tooling. It does not convert Hugging Face checkpoints.

## Reference pins

- Model repository: `Qwen/Qwen3.8-Flash-Next`
- Package-tooling revision: `f5d08274bafd880402bd16f5e3e6c514136ec06c`
- Verified model head on 2026-08-28: `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- The two model revisions differ only in `README.md`. Config, tokenizer and
  template assets, tensor index, and weights are unchanged.
- Semantic reference: Transformers `v5.16.0`, commit
  `93d1bcfbd2af5798e2f66bf7955e31a537902b64`.

The authoritative producer contract is
`../precisit-gemma4/SMT-QWEN4EXP-MLX-RUNTIME-CONTRACT.md`. The sibling source
tree is named `Sources/`, not `Source/`.

## Artifact and config

The package is a standard MLX-LM directory with safetensor shards, tokenizer
assets, optional generation and chat-template assets, and conversion or
quantization manifests.

The producer flattens the upstream nested `text_config` without changing its
number formatting. The emitted root `model_type` is `qwen4_exp_text`. All text
runtime fields are at the root. A quantized package has identical
`quantization` and `quantization_config` dictionaries. Their root entry gives
the default MLX affine `{group_size, bits, mode}` and path entries override it.

The full model uses 48 layers: 36 `linear_attention` layers and 12
`full_attention` entries. At runtime, `full_attention` means Qwen Sparse
Attention. PLE layer IDs are one-indexed.

## Tensor partition and layout

The producer applies one rename:

```text
model.language_model.* -> model.*
```

It retains `lm_head.*` and `mtp.*`. It removes only `model.visual.*` and the
vision/video processor assets. It does not transpose, split, fuse, reorder, or
otherwise transform retained tensor payloads.

Important full-model tensors are:

| Family | Name | Shape and layout |
|---|---|---|
| Token embedding | `model.embed_tokens.weight` | `[248320, 2560]` |
| Output head | `lm_head.weight` | `[248320, 2560]`; untied |
| DeltaNet QKV | `model.layers.N.linear_attn.in_proj_qkv.weight` | `[10240, 2560]`; rows are q(2048), k(2048), v(6144) |
| DeltaNet convolution | `model.layers.N.linear_attn.conv1d.weight` | `[10240, 1, 4]`; depthwise |
| DeltaNet state scalars | `model.layers.N.linear_attn.A_log`, `dt_bias` | `[48]`; compute in FP32 |
| QSA query and gate | `model.layers.N.self_attn.q_proj.weight` | `[12288, 2560]`; each of 24 heads stores q(256), gate(256) |
| QSA indexer | `model.layers.N.self_attn.indexer.index_qk_proj.weight` | `[640, 2560]`; 4 query heads then 1 key head, each width 128 |
| Expert gate/up | `model.layers.N.mlp.experts.gate_up_proj` | `[512, 1280, 2560]`; no `.weight`; each expert stores gate(640), up(640) |
| Expert down | `model.layers.N.mlp.experts.down_proj` | `[512, 2560, 640]`; no `.weight` |
| Router | `model.layers.N.mlp.gate.weight` | `[512, 2560]` |
| Shared expert | `model.layers.N.mlp.shared_expert.{gate,up,down}_proj.weight` | standard linear layouts |
| Shared gate | `model.layers.N.mlp.shared_expert_gate.weight` | `[1, 2560]` |
| Hyper-connection norm | `model.layers.N.{attn,mlp}_hyper_connection.hc_norm.weight` | `[10240]` |
| Hyper-connection mixers | `...input_mix_weight_down.weight`, `...input_mix_weight_up.weight` | `[320, 10240]`, `[10240, 320]` |
| Hyper-connection injection | `...block_inject_weight.weight` | `[4, 10240]` |
| Final mixer | `model.hyper_connection_mixer.*` | same norm and input-mixer layout; there is no `model.norm.weight` |
| N-gram table | `model.layers.1.ple.ple_embedding.ngram_embedding.shard_S.weight` | 128 independent `[2500012, 160]` shards |
| N-gram buffers | `...ple_embedding.{ngram_heads_vocab_sizes,ngram_heads_offsets,layer_multipliers}` | I64 `[16]`, `[16]`, `[3]` |
| PLE projections | `model.layers.1.ple.{key_proj,value_proj}.weight` | source linear layouts |
| PLE convolution | `model.layers.1.ple.conv1d.weight` | `[10240, 1, 4]`; depthwise and dilated at runtime |

All `Qwen4ExpTextRMSNorm` weights are zero-centered and use `1 + weight`.
`linear_attn.norm` is the exception: it is an ordinary RMSNorm gain followed
by the configured gate activation. There is no embedding scale.

## Quantization

Quantization uses MLX affine packing along the last axis and preserves rank.
For a quantized `.weight` module, the package stores `weight`, `scales`, and
`biases`. The expert arrays have parameter-scoped triplets:

```text
experts.gate_up_proj
experts.gate_up_proj.scales
experts.gate_up_proj.biases
experts.down_proj
experts.down_proj.scales
experts.down_proj.biases
```

The runtime treats a parameter as quantized only when its `scales` sibling is
present. Packed expert gate/up rows may be sliced at the intermediate-size
boundary because groups do not cross rows.

The n-gram table remains split. Each shard is an independent quantized
embedding with 4-bit, group-size 32 affine data in the production recipe. The
runtime gathers only requested rows and never materializes or dequantizes the
full table. Token embeddings and the output head use their own configured
precision. Protected gates, routers, indexers, rank-one tensors, convolutions,
and I64 buffers remain unquantized when no `scales` tensor exists.

The runtime derives the shard row count from `shard_0.weight` before model
construction. It then verifies that all hash-buffer offsets fit inside the
combined shard table. It does not wrap or clamp global n-gram IDs.

## MTP and supported scope

The producer retains `mtp.*`, including its decoder layer and expert stacks.
Ordinary autoregressive generation does not use these tensors. The text v1
runtime removes `mtp.*` during load and never requires them. Packages without
MTP tensors are also valid.

Vision, video, image processing, and multimodal position handling are out of
scope. Passing multimodal inputs is unsupported.

## Current tiny-fixture gap

The current producer tiny artifacts have four 64-row n-gram shards, for 256
rows total. Their hash buffers require 272 rows. The tokenizer is also the
documented stub. These artifacts cannot satisfy ordinary `mlx_lm.load` and
reference-generation tests until the producer regenerates them with a valid
table and pinned tokenizer assets. The runtime reports the table mismatch
explicitly instead of guessing a different hashing rule.

## Host-memory classification (2026-10-17, from precisit/modelopt-to-mlx Stage 2)

**Unsafe on a 128 GB host: Qwen3.8-Flash-Next with the PLE n-gram table resident.** On altair
(M5 Max, 128 GB, no swap), this runtime with the NVIDIA FP4-Local pack and a resident PLE
(26.8 GiB) peaked at **117.7 GiB** of MLX memory on a 2 k-token prompt. The lab budget after a
watchdog panic on that host is a ≤ 105 GiB whole-machine footprint (hq-infra). Above it, macOS
can hang until the watchdog reboots it: wired and anonymous memory cannot be paged without swap.
Do not run this configuration on a 128 GB machine. It needs an mmap/offloaded PLE path (an
upstream candidate, oMLX has one) or a larger host. Evidence: modelopt-to-mlx
`stage2/receipts/s2-fingerprint-fork.json`, `stage2/PROTOCOL.md` amendment 5,
generic-dev-docs lesson 7.
