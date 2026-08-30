# Qwen4-Exp runtime implementation note

The semantic reference is Transformers 5.16.0 at commit
`93d1bcfbd2af5798e2f66bf7955e31a537902b64`. The model package is pinned to
`Qwen/Qwen3.8-Flash-Next` revision
`f5d08274bafd880402bd16f5e3e6c514136ec06c`.

## Supported scope

The runtime supports text-only ordinary autoregressive generation. It loads the
flat `qwen4_exp_text` package emitted by the sibling `precisit-gemma4` tooling.
It implements Gated DeltaNet, Qwen Sparse Attention, MoE with a shared expert,
gated residual streams, PLE and sharded n-gram embeddings, and hybrid caches.
It accepts packages with or without MTP tensors and ignores MTP during ordinary
generation.

QSA uses a full unquantized KV cache. Quantized, rotating, and batched padded
QSA caches fail explicitly because their index space differs from the retained
raw-key indexer state.

Vision, video, image processing, and multimodal position handling are not
supported. The runtime does not use or change `mlx_lm.convert`.

## Package contract

The complete contract is in `QWEN4_EXP_RUNTIME_CONTRACT.md`. The runtime expects
the producer's source tensor layouts, including `[out, 1, kernel]` depthwise
convolutions, fused suffix-less expert parameters, rank-preserving MLX affine
quantization, and independently quantized n-gram shards. Load-time sanitation
only adapts these fixed layouts to existing MLX modules. It does not ingest or
convert a Hugging Face checkpoint.

## Validation status

The legal synthetic package matches the pinned Transformers reference within
`3.8e-8` maximum absolute logit error. Its segmented cache test includes prompt
prefill, one-token decode, and multi-token decode. The focused suite also checks
QSA boundaries, DeltaNet state, MoE routing and shared-expert output, n-gram EOS
boundaries, chat-template modes, quantized linear and embedding loads, and a
bounded `mlx_lm.generate` smoke.

The current sibling tiny artifact cannot be a runtime golden yet. Its four
n-gram shards contain 256 rows, but its hash buffers address 272 rows. Its
tokenizer is also a documented stub. The runtime rejects the invalid table
instead of wrapping indices or guessing a layout. The producer fixture must be
regenerated with at least 272 table rows and real pinned tokenizer assets.

The 329.6 GiB BF16 text package still needs a real-weight parity run on a
large-memory host. `benchmarks/qwen4_exp_parity.py` and its adjacent manifest
run the pinned Transformers and MLX stages separately so both models do not
need to be resident at the same time. The cross-backend tolerance in the
manifest is provisional until that run is complete.
