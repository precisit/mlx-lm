import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from mlx_lm import generate, load
from mlx_lm.models.cache import ArraysCache, CacheList, QuantizedKVCache
from mlx_lm.models.qwen4_exp import (
    Model,
    ModelArgs,
    Qwen4ExpGatedDeltaNet,
    Qwen4ExpNGramEmbedding,
    Qwen4ExpSparseAttention,
    Qwen4ExpSparseMoeBlock,
    _qsa_select_tokens,
)
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
from mlx_lm.utils import load_model


def tiny_config():
    return {
        "model_type": "qwen4_exp_text",
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "partial_rotary_factor": 0.25,
        "rope_theta": 10_000.0,
        "max_position_embeddings": 128,
        "indexer_n_heads": 1,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 8,
        "indexer_budget": 2,
        "indexer_compress_ratio": 2,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "output_gate_type": "sigmoid",
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "norm_topk_prob": True,
        "ngram_size": 3,
        "heads_per_ngram": 1,
        "split_ngram_parts": 2,
        "ngram_shard_rows": 12,
        "ple_layer_ids": [2],
        "ple_embed_dim": 64,
        "ple_conv_kernel_size": 4,
        "hc_count": 2,
        "hc_lowrank": 8,
        "tie_word_embeddings": False,
        "bos_token_id": 0,
        "eos_token_id": 1,
    }


def _values(name, shape):
    size = 1
    for dim in shape:
        size *= dim
    seed = sum(name.encode("utf-8")) % 37
    values = (mx.arange(size, dtype=mx.float32) + seed) % 41
    return ((values - 20) / 1024).reshape(shape)


def producer_weights(config):
    model = Model(ModelArgs.from_dict(config))
    weights = {}
    for name, value in tree_flatten(model.parameters()):
        if name.endswith("ngram_heads_vocab_sizes"):
            weights[name] = mx.array([11, 13], dtype=mx.int64)
        elif name.endswith("ngram_heads_offsets"):
            weights[name] = mx.array([0, 11], dtype=mx.int64)
        elif name.endswith("layer_multipliers"):
            weights[name] = mx.array([17, 29, 43], dtype=mx.int64)
        elif name.endswith("A_log"):
            weights[name] = mx.full(value.shape, -0.25)
        elif name.endswith("dt_bias"):
            weights[name] = mx.full(value.shape, 0.1)
        elif name.endswith("linear_attn.norm.weight"):
            weights[name] = mx.ones(value.shape)
        else:
            weights[name] = _values(name, value.shape)

    for layer_index in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer_index}.mlp"
        gate = weights.pop(f"{prefix}.switch_mlp.gate_proj.weight")
        up = weights.pop(f"{prefix}.switch_mlp.up_proj.weight")
        weights[f"{prefix}.experts.gate_up_proj"] = mx.concatenate(
            [gate, up], axis=-2
        )
        weights[f"{prefix}.experts.down_proj"] = weights.pop(
            f"{prefix}.switch_mlp.down_proj.weight"
        )

    for name, value in list(weights.items()):
        if name.endswith("conv1d.weight"):
            weights[name] = value.moveaxis(1, 2)
    return weights


def quantize_weights(weights):
    quantized = {}
    for name, value in weights.items():
        if (
            mx.issubdtype(value.dtype, mx.floating)
            and value.ndim >= 2
            and value.shape[-1] % 32 == 0
            and not name.endswith("conv1d.weight")
        ):
            group_size = 32
            weight, scales, biases = mx.quantize(
                value, group_size=group_size, bits=4, mode="affine"
            )
            quantized[name] = weight
            if name.endswith(".weight"):
                prefix = name[: -len(".weight")]
                quantized[f"{prefix}.scales"] = scales
                quantized[f"{prefix}.biases"] = biases
            else:
                quantized[f"{name}.scales"] = scales
                quantized[f"{name}.biases"] = biases
        else:
            quantized[name] = value
    return quantized


def write_tokenizer(path):
    vocab = {"<bos>": 0, "<eos>": 1, "<unk>": 2, "<pad>": 3}
    vocab.update({f"t{index}": index for index in range(4, 32)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path / "tokenizer.json"))
    chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n"
        "{% if enable_thinking is defined and enable_thinking is false %}"
        "<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}{% endif %}"
    )
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "chat_template": chat_template,
    }
    (path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))


def write_package(path, quantized=False):
    config = tiny_config()
    weights = producer_weights(config)
    if quantized:
        weights = quantize_weights(weights)
        quantization = {"group_size": 32, "bits": 4, "mode": "affine"}
        for shard_index in range(config["split_ngram_parts"]):
            quantization[
                "model.layers.1.ple.ple_embedding.ngram_embedding."
                f"shard_{shard_index}"
            ] = {"group_size": 32, "bits": 4, "mode": "affine"}
        config["quantization"] = quantization
        config["quantization_config"] = quantization
    config.pop("ngram_shard_rows")
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    write_tokenizer(path)
    mx.save_safetensors(str(path / "model.safetensors"), weights)


class TestQwen4Exp(unittest.TestCase):
    def test_qsa_block_boundaries(self):
        scores = mx.array(
            [
                [
                    [9.0, 1.0, 2.0],
                    [9.0, 1.0, 2.0],
                    [1.0, 9.0, 2.0],
                    [1.0, 9.0, 2.0],
                    [1.0, 2.0, 9.0],
                    [1.0, 2.0, 9.0],
                ]
            ]
        )
        indices, valid = _qsa_select_tokens(
            scores, mx.arange(6), compress_ratio=2, block_topk=1
        )
        indices = indices.tolist()[0]
        valid = valid.tolist()[0]
        selected = [
            {token for token, keep in zip(row, row_valid) if keep}
            for row, row_valid in zip(indices, valid)
        ]
        self.assertEqual(selected[0], {0})
        self.assertEqual(selected[1], {0, 1})
        self.assertEqual(selected[2], {0, 1, 2})
        self.assertEqual(selected[3], {2, 3})
        self.assertEqual(selected[4], {2, 3, 4})
        self.assertEqual(selected[5], {4, 5})

        scores = mx.broadcast_to(mx.array([[[1.0, 9.0, 2.0]]]), (1, 9, 3))
        indices, valid = _qsa_select_tokens(
            scores, mx.arange(9), compress_ratio=4, block_topk=1
        )
        selected = [
            {token for token, keep in zip(row, row_valid) if keep}
            for row, row_valid in zip(indices.tolist()[0], valid.tolist()[0])
        ]
        self.assertEqual(selected[0], {0})
        self.assertEqual(selected[3], {0, 1, 2, 3})
        self.assertEqual(selected[4], {0, 1, 2, 3, 4})
        self.assertEqual(selected[7], {4, 5, 6, 7})
        self.assertEqual(selected[8], {4, 5, 6, 7, 8})

    def test_delta_state_matches_full_sequence(self):
        args = ModelArgs.from_dict(tiny_config())
        delta = Qwen4ExpGatedDeltaNet(args)
        inputs = _values("delta-input", (1, 7, args.hidden_size))
        full = delta(inputs)
        cache = ArraysCache(2)
        cached = mx.concatenate(
            [delta(inputs[:, index : index + 1], cache=cache) for index in range(7)],
            axis=1,
        )
        mx.eval(full, cached)
        self.assertTrue(mx.allclose(full, cached, atol=2e-5, rtol=2e-5))
        self.assertEqual(cache[0].shape[1], args.linear_conv_kernel_dim - 1)
        self.assertIsNotNone(cache[1])

    def test_qsa_rejects_unsupported_cache_mode(self):
        attention = Qwen4ExpSparseAttention(ModelArgs.from_dict(tiny_config()))
        cache = CacheList(QuantizedKVCache(), ArraysCache(1))
        with self.assertRaisesRegex(ValueError, "unquantized full KV cache"):
            attention(mx.zeros((1, 1, 32)), cache)

    def test_moe_shared_expert_contribution(self):
        args = ModelArgs.from_dict(tiny_config())
        moe = Qwen4ExpSparseMoeBlock(args)
        moe.gate.weight = mx.array(
            [[1.0] + [0.0] * 31, [-1.0] + [0.0] * 31]
        )
        for projection in (
            moe.switch_mlp.gate_proj,
            moe.switch_mlp.up_proj,
            moe.switch_mlp.down_proj,
        ):
            projection.weight = mx.zeros_like(projection.weight)
        moe.shared_expert.gate_proj.weight = mx.eye(16, 32)
        moe.shared_expert.up_proj.weight = mx.eye(16, 32)
        moe.shared_expert.down_proj.weight = mx.eye(32, 16)
        moe.shared_expert_gate.weight = mx.zeros((1, 32))
        inputs = mx.ones((1, 1, 32))
        output = moe(inputs)
        expected = 0.5 * moe.shared_expert(inputs)
        mx.eval(output, expected)
        self.assertTrue(mx.allclose(output, expected, atol=1e-6, rtol=1e-6))

    def test_moe_topk_routing_and_normalization(self):
        config = tiny_config()
        config["num_experts"] = 3
        config["num_experts_per_tok"] = 2
        moe = Qwen4ExpSparseMoeBlock(ModelArgs.from_dict(config))
        moe.gate.weight = mx.zeros_like(moe.gate.weight)
        moe.gate.weight = moe.gate.weight.at[:, 0].add(mx.array([0.0, 1.0, 2.0]))
        for projection in (
            moe.switch_mlp.gate_proj,
            moe.switch_mlp.up_proj,
            moe.switch_mlp.down_proj,
            moe.shared_expert.gate_proj,
            moe.shared_expert.up_proj,
            moe.shared_expert.down_proj,
        ):
            projection.weight = mx.zeros_like(projection.weight)
        moe.switch_mlp.gate_proj.weight = (
            moe.switch_mlp.gate_proj.weight.at[:, 0, 0].add(1.0)
        )
        moe.switch_mlp.up_proj.weight = (
            moe.switch_mlp.up_proj.weight.at[:, 0, 0].add(
                mx.array([1.0, 2.0, 3.0])
            )
        )
        moe.switch_mlp.down_proj.weight = (
            moe.switch_mlp.down_proj.weight.at[:, 0, 0].add(1.0)
        )

        inputs = mx.zeros((1, 1, 32)).at[..., 0].add(1.0)
        output = moe(inputs)
        probabilities = mx.softmax(mx.array([0.0, 1.0, 2.0]), precise=True)
        scores = probabilities[1:] / probabilities[1:].sum()
        expected = nn.silu(mx.array(1.0)) * (
            2.0 * scores[0] + 3.0 * scores[1]
        )
        mx.eval(output, expected)
        self.assertTrue(mx.allclose(output[0, 0, 0], expected, atol=1e-6))
        self.assertTrue(mx.all(output[0, 0, 1:] == 0))

    def test_ngram_eos_boundaries_and_cache(self):
        args = ModelArgs.from_dict(tiny_config())
        ngram = Qwen4ExpNGramEmbedding(args)
        ngram.ngram_heads_vocab_sizes = mx.array([11, 13], dtype=mx.int64)
        ngram.ngram_heads_offsets = mx.array([0, 11], dtype=mx.int64)
        ngram.layer_multipliers = mx.array([17, 29, 43], dtype=mx.int64)
        for shard_index in range(2):
            start = shard_index * 12
            rows = mx.arange(start, start + 12, dtype=mx.float32)
            getattr(ngram.ngram_embedding, f"shard_{shard_index}").weight = (
                mx.broadcast_to(rows[:, None], (12, 32))
            )
        tokens = mx.array([[4, 5, 1, 6, 7, 8]], dtype=mx.int32)
        full = ngram(tokens, None)
        cache = ArraysCache(4)
        cached = mx.concatenate(
            [ngram(tokens[:, index : index + 1], cache) for index in range(6)],
            axis=1,
        )
        mx.eval(full, cached)
        self.assertTrue(mx.array_equal(full, cached))
        ids = full[0].reshape(6, 2, 32)[:, :, 0]
        expected_ids = mx.array(
            [[1, 21], [0, 21], [7, 16], [2, 13], [8, 19], [1, 20]]
        )
        self.assertTrue(mx.array_equal(ids, expected_ids))

    def test_unquantized_load_reference_logits_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "bf16"
            write_package(package)
            model, _ = load_model(package, lazy=True)
            tokens = mx.array([[4, 5, 6, 1, 7, 8, 9]], dtype=mx.int32)
            full = model(tokens)
            cache = model.make_cache()
            cached = mx.concatenate(
                [
                    model(tokens[:, :4], cache=cache),
                    model(tokens[:, 4:5], cache=cache),
                    model(tokens[:, 5:7], cache=cache),
                ],
                axis=1,
            )
            mx.eval(full, cached)
            self.assertTrue(mx.allclose(full, cached, atol=3e-5, rtol=3e-5))
            # Transformers v5.16.0 at commit 93d1bcfbd2af5798.
            expected = mx.array(
                [
                    [
                        -0.097280949,
                        -0.039801262,
                        0.126807049,
                        0.045711346,
                        -0.091071919,
                        -0.086948529,
                    ],
                    [
                        -0.001456843,
                        -0.071175970,
                        -0.049628258,
                        0.095804974,
                        0.017247541,
                        -0.044567328,
                    ],
                    [
                        0.116401464,
                        -0.035904929,
                        -0.103702828,
                        -0.029910080,
                        0.115518592,
                        0.064229719,
                    ],
                    [
                        0.094933897,
                        -0.037663944,
                        -0.087214634,
                        -0.022441469,
                        0.121550217,
                        0.039282702,
                    ],
                    [
                        0.014137117,
                        0.122665271,
                        -0.039631113,
                        -0.075094469,
                        -0.001041869,
                        0.071327463,
                    ],
                    [
                        -0.069635332,
                        0.007744635,
                        0.121027857,
                        -0.022068745,
                        -0.068561390,
                        -0.043378122,
                    ],
                    [
                        -0.075309515,
                        -0.059229486,
                        0.045862600,
                        0.130260542,
                        -0.063324772,
                        -0.086109549,
                    ],
                ]
            )
            self.assertTrue(
                mx.allclose(full[0, :, :6], expected, atol=5e-5, rtol=5e-5)
            )
            self.assertTrue(
                mx.allclose(cached[0, :, :6], expected, atol=5e-5, rtol=5e-5)
            )

    def test_quantized_load_chat_template_and_generate(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "quantized"
            write_package(package, quantized=True)
            model, tokenizer = load(str(package), lazy=True)
            self.assertIsInstance(model.model.embed_tokens, nn.QuantizedEmbedding)
            shard = model.layers[1].ple.ple_embedding.ngram_embedding.shard_0
            self.assertIsInstance(shard, nn.QuantizedEmbedding)
            self.assertIsInstance(
                model.layers[0].mlp.switch_mlp.gate_proj,
                QuantizedSwitchLinear,
            )
            messages = [{"role": "user", "content": "t4"}]
            thinking = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            direct = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            self.assertTrue(thinking.endswith("<think>\n"))
            self.assertTrue(direct.endswith("<think>\n\n</think>\n\n"))
            tokens = mx.array([[4, 5, 6, 1, 7, 8]], dtype=mx.int32)
            full = model(tokens)
            cache = model.make_cache()
            cached = mx.concatenate(
                [
                    model(tokens[:, :3], cache=cache),
                    model(tokens[:, 3:4], cache=cache),
                    model(tokens[:, 4:], cache=cache),
                ],
                axis=1,
            )
            mx.eval(full, cached)
            self.assertTrue(mx.allclose(full, cached, atol=5e-4, rtol=5e-4))
            output = generate(model, tokenizer, [4, 5], max_tokens=2)
            self.assertIsInstance(output, str)

    def test_invalid_ngram_package_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "invalid"
            config = tiny_config()
            weights = producer_weights(config)
            for shard_index in range(2):
                key = (
                    "model.layers.1.ple.ple_embedding.ngram_embedding."
                    f"shard_{shard_index}.weight"
                )
                weights[key] = weights[key][:-1]
            config.pop("ngram_shard_rows")
            package.mkdir()
            (package / "config.json").write_text(json.dumps(config))
            mx.save_safetensors(str(package / "model.safetensors"), weights)
            with self.assertRaisesRegex(ValueError, "hash buffers require 24"):
                load_model(package, lazy=True)


if __name__ == "__main__":
    unittest.main()
