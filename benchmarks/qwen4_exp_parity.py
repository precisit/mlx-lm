#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TRANSFORMERS_VERSION = "5.16.0"


def read_manifest(path: Path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def token_chunks(token_ids, cache_config):
    prefill = cache_config["prefill_length"]
    chunks = [token_ids[:prefill]]
    offset = prefill
    for length in cache_config["decode_chunks"]:
        chunks.append(token_ids[offset : offset + length])
        offset += length
    if offset != len(token_ids) or any(len(chunk) == 0 for chunk in chunks):
        raise ValueError("The cache chunks must cover all input IDs exactly.")
    return chunks


def metrics(actual, expected, atol, rtol):
    delta = np.abs(actual - expected)
    limit = atol + rtol * np.abs(expected)
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "passed": bool(np.all(delta <= limit)),
    }


def run_reference(args, manifest, manifest_hash):
    import torch
    import transformers
    from transformers.models.qwen4_exp import Qwen4ExpForConditionalGeneration

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Transformers {TRANSFORMERS_VERSION} is required; found "
            f"{transformers.__version__}."
        )

    model_config = manifest["model"]
    model = Qwen4ExpForConditionalGeneration.from_pretrained(
        model_config["repository"],
        revision=model_config["revision"],
        dtype=torch.bfloat16,
        device_map=args.device_map,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    )
    model.eval()
    token_ids = manifest["input_ids"]
    chunks = token_chunks(token_ids, manifest["cache"])
    input_device = model.get_input_embeddings().weight.device

    with torch.no_grad():
        tokens = torch.tensor([token_ids], dtype=torch.long, device=input_device)
        full = model(input_ids=tokens, use_cache=False).logits.float().cpu().numpy()

        cached_parts = []
        cache = None
        for chunk in chunks:
            tokens = torch.tensor([chunk], dtype=torch.long, device=input_device)
            output = model(
                input_ids=tokens,
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            cached_parts.append(output.logits.float().cpu().numpy())
        cached = np.concatenate(cached_parts, axis=1)

    tolerance = manifest["tolerances"]["reference_cache"]
    result = metrics(full, cached, tolerance["atol"], tolerance["rtol"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "manifest_sha256": manifest_hash,
        "model_revision": model_config["revision"],
        "transformers_version": transformers.__version__,
        "reference_cache": result,
    }
    np.savez(
        output_path,
        input_ids=np.asarray(token_ids, dtype=np.int64),
        full_logits=full,
        cached_logits=cached,
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(json.dumps(metadata, indent=2))
    if not result["passed"]:
        raise AssertionError("Transformers cached logits do not match full logits.")


def run_mlx(args, manifest, manifest_hash):
    import mlx.core as mx

    from mlx_lm import load

    reference = np.load(args.reference_output)
    metadata = json.loads(reference["metadata"].item())
    if metadata["manifest_sha256"] != manifest_hash:
        raise ValueError("The reference output uses a different manifest.")

    token_ids = manifest["input_ids"]
    if not np.array_equal(reference["input_ids"], token_ids):
        raise ValueError("The reference input IDs do not match the manifest.")

    model, _ = load(args.package, lazy=True)
    tokens = mx.array([token_ids], dtype=mx.int32)
    full = model(tokens)
    cache = model.make_cache()
    cached_parts = []
    for chunk in token_chunks(token_ids, manifest["cache"]):
        cached_parts.append(
            model(mx.array([chunk], dtype=mx.int32), cache=cache)
        )
    cached = mx.concatenate(cached_parts, axis=1)
    mx.eval(full, cached)
    full = np.asarray(full.astype(mx.float32))
    cached = np.asarray(cached.astype(mx.float32))

    cache_tolerance = manifest["tolerances"]["mlx_cache"]
    reference_tolerance = manifest["tolerances"]["mlx_reference"]
    results = {
        "mlx_cache": metrics(
            full,
            cached,
            cache_tolerance["atol"],
            cache_tolerance["rtol"],
        ),
        "mlx_full_reference": metrics(
            full,
            reference["full_logits"],
            reference_tolerance["atol"],
            reference_tolerance["rtol"],
        ),
        "mlx_cached_reference": metrics(
            cached,
            reference["cached_logits"],
            reference_tolerance["atol"],
            reference_tolerance["rtol"],
        ),
    }
    print(json.dumps(results, indent=2))
    if not all(result["passed"] for result in results.values()):
        raise AssertionError("One or more Qwen4-Exp parity checks failed.")


def main():
    parser = argparse.ArgumentParser(
        description="Run two-stage Qwen4-Exp real-weight parity."
    )
    parser.add_argument("backend", choices=("reference", "mlx"))
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).with_suffix(".json")),
    )
    parser.add_argument("--output")
    parser.add_argument("--package")
    parser.add_argument("--reference-output")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    manifest, manifest_hash = read_manifest(Path(args.manifest))
    if args.backend == "reference":
        if not args.output:
            parser.error("reference requires --output")
        run_reference(args, manifest, manifest_hash)
    else:
        if not args.package or not args.reference_output:
            parser.error("mlx requires --package and --reference-output")
        run_mlx(args, manifest, manifest_hash)


if __name__ == "__main__":
    main()
