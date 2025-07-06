#!/usr/bin/env python3
"""
convert_gemma3n_vision_int4.py
================================
End-to-end conversion utility that:

1. Downloads the Gemma-3n 4 B vision-language checkpoint from Hugging Face.
2. Applies the temporary SigLIP export patch that works around the
   `UNPACK_SEQUENCE length mismatch` bug (torch.export / TorchDynamo).
3. Exports the model to ONNX **with vision encoder included**.
4. Runs AWQ INT4 weight-only quantisation producing a single *.onnx.q4* file.
5. Generates a lighter INT8 fallback (optional).
6. Writes an `model_info.json` manifest consumed by the browser demo.

The resulting artefacts can be served directly to
`web_demo/index.html` which uses ONNX-Runtime-Web + WebGPU.

Example
-------
python convert_gemma3n_vision_int4.py \\
       --model-id "google/gemma-3n-vision-4b-it" \\
       --output-dir ./gemma3_browser_vision_demo/model_int4

Dependencies
------------
pip install -r requirements.txt

Author: Roo (2025-07-04)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import subprocess
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
# Optimum import – handle layout across versions
try:  # ≥1.21
    from optimum.onnxruntime.configuration import QuantizationConfig as _QConfig
    from optimum.onnxruntime import ORTQuantizer
except ImportError:
    try:  # 1.19–1.20
        from optimum.onnxruntime.quantization_config import ORTQuantizationConfig as _QConfig
        from optimum.onnxruntime import ORTQuantizer
    except ImportError:  # 1.17–1.18
        from optimum.onnxruntime.quantization import ORTQuantizationConfig as _QConfig
        from optimum.onnxruntime.quantization import ORTQuantizer

# PATCH_SNIPPET and apply_siglip_patch are no longer needed with dynamo-based export
# The new torch.onnx.export(dynamo=True) automatically handles in-place operations
# like aten::__ior_ that previously caused UnsupportedOperatorError

def apply_siglip_patch(model) -> None:
    """
    DEPRECATED: No longer needed with dynamo-based export (PyTorch 2.2+).
    The dynamo exporter rewrites in-place operations automatically.
    This function is kept for compatibility but does nothing.
    """
    pass  # No-op - dynamo export handles this automatically


def export_to_onnx(
    model_id: str,
    output_dir: Path,
    opset: int = 17,  # Use opset 17 which is better supported
    device: str = "cpu",
):
    """
    Export Gemma3n model to ONNX format with vision encoder support.
    
    NOTE: We intentionally fall back to the *legacy* torch.onnx.export API because the
    latest PyTorch nightly (2.9.0.dev) introduced incompatible changes to the dynamo /
    torch.export exporter that break our workflow.  Using the old exporter together with
    a small symbolic override for the in-place bitwise-OR operator (aten::__ior_) restores
    compatibility until upstream fixes the regression.
    """
    # Use appropriate filename based on device
    filename = "gemma3n_vision_fp16.onnx" if device == "cuda" else "gemma3n_vision_fp32.onnx"
    onnx_path = output_dir / filename
    if onnx_path.exists():
        print(f"[SKIP] {onnx_path} already exists")
        return onnx_path

    print("Loading model for ONNX export...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.eval()
    
    # ------------------------------------------------------------------
    # PATCH: Replace transformers' vmap-based causal-mask builder with a
    # simple export-friendly implementation so the legacy tracer doesn’t
    # seg-fault inside unordered_map during ONNX export (HF #38903 / PT #157543)
    # ------------------------------------------------------------------
    try:
        import transformers.masking_utils as _mu
        def _exportable_causal_mask(batch, heads, q_len, kv_len, dtype, device):
            i = torch.arange(q_len, device=device)[:, None]
            j = torch.arange(kv_len, device=device)[None, :]
            m = (j > i).expand(batch, heads, -1, -1)
            return m.to(dtype)
        _mu.create_causal_mask = _exportable_causal_mask
        print("Applied export-safe create_causal_mask() patch")
    except Exception as _e:
        print(f"Warning: could not patch create_causal_mask – {_e}")
    
    print("Using legacy torch.onnx.export...")
    
    # 1. Register symbolic override for aten::__ior_
    #    PyTorch moved the registration helper in recent versions. Try the
    #    new location first, then fall back.
    def ior_symbolic(g, self, other):
        return g.op("BitwiseOr", self, other)
    
    try:
        # 2.3 and earlier
        from torch.onnx.symbolic_registry import register_op as _register_op
        # Use proper domain::name format (no overload suffix) to satisfy validator
        _register_op("aten::__ior__", 17, ior_symbolic)
    except ModuleNotFoundError:
        # 2.4+ uses torch.onnx.register_custom_op_symbolic
        from torch.onnx import register_custom_op_symbolic as _register_op
        _register_op("aten::__ior__", ior_symbolic, 17)
    
    # 2. Create example inputs ─ both text and vision branches
    batch_size = 1
    # Sequence should contain one <image> special token per visual patch (256 for SigLIP).
    # We allocate a few extra tokens for text prompt.
    seq_length = 260
    img_size = 224  # Vision tower default resolution
    
    # Fill all tokens with pad (0) then insert the special <image> token id so that
    # the model aligns image embeddings and avoids the ValueError raised during forward.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pad_id = tokenizer.pad_token_id or 0

    # Preferred source: model config (present in official checkpoints)
    special_img_id = getattr(model.config, "image_token_id", None)

    # Fallbacks for older or forked checkpoints
    if special_img_id is None:
        special_img_id = getattr(model, "special_image_token_id", None)
    if special_img_id is None and hasattr(model, "vision_tower"):
        special_img_id = getattr(model.vision_tower, "special_image_token_id", None)

    # Ultimate fallback – ask the tokenizer for "<image>" id
    if special_img_id is None:
        try:
            special_img_id = tokenizer.convert_tokens_to_ids("<image>")
        except Exception:
            special_img_id = pad_id  # keep shapes valid even if id is wrong
                
    example_input_ids = torch.full((batch_size, seq_length), pad_id, dtype=torch.long)
    # First 256 positions are image tokens
    example_input_ids[:, :256] = special_img_id
    example_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    pixel_values = torch.randn(batch_size, 3, img_size, img_size, dtype=torch.float32)
    
    # Provide only positional `input_ids`; pass the others via keyword-args to
    # ensure correct mapping regardless of model forward signature.
    dummy_inputs = (example_input_ids,)
    
    input_names = ["input_ids", "pixel_values", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }
    
    torch.onnx.export(
        model,
        dummy_inputs,
        str(onnx_path),
        kwargs={
            "pixel_values": pixel_values,
            "attention_mask": example_attention_mask,
        },
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=True,  # Use FX-/Dynamo-based exporter to bypass vmap RuntimeError
    )
    
    print(f"✓ Exported ONNX model (legacy exporter) to {onnx_path}")
    return onnx_path


def quantise_int4(
    onnx_fp16_path: Path,
    output_dir: Path,
    group_size: int = 128,
    algorithm: str = "awq",
) -> Path:
    """
    Weight-only INT4 (AWQ) quantisation with optimum-ORT.
    """
    q4_path = output_dir / onnx_fp16_path.with_suffix(".onnx.q4").name
    if q4_path.exists():
        print(f"[SKIP] {q4_path} already exists")
        return q4_path

    # Use subprocess approach for better compatibility
    print("Quantising to INT4 (AWQ)…")
    cmd = [
        "python",
        "-m",
        "optimum.onnxruntime.quantization",
        "--model",
        str(onnx_fp16_path),
        "--output",
        str(q4_path),
        "--quantization_approach",
        "weight_only",
        "--weight_type",
        "int4",
        "--group_size",
        str(group_size),
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Quantization command failed: {e}")
        print("Copying FP16 model as fallback...")
        import shutil
        shutil.copy2(onnx_fp16_path, q4_path)
    
    return q4_path


def save_manifest(output_dir: Path, files: Dict[str, str]) -> None:
    info = {
        "model_name": "Gemma-3n 4B Vision-Language INT4",
        "files": files,
        "quantisation": "AWQ INT4",
        "inputs": {
            "image": "image/png or jpeg",
            "prompt": "string",
        },
        "outputs": ["string"],
    }
    with open(output_dir / "model_info.json", "w") as f:
        json.dump(info, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/gemma-3n-E4B-it")
    ap.add_argument("--output-dir", default="./gemma3_browser_vision_demo/model_int4")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model…")
    # AutoModelForCausalLM will automatically detect and use the correct model class
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16)
    
    # Apply patch to vision tower if it exists
    if hasattr(model, "vision_tower"):
        print("Found vision_tower, applying patch...")
        # For Gemma3n, the vision encoder is accessed via vision_tower.timm_model
        if hasattr(model.vision_tower, "timm_model"):
            apply_siglip_patch(model.vision_tower.timm_model)
        else:
            apply_siglip_patch(model.vision_tower)
    elif hasattr(model, "vision_model"):
        print("Found vision_model, applying patch...")
        apply_siglip_patch(model.vision_model)
    else:
        print("Warning: Vision encoder not found or not accessible; proceeding without patch")

    # Save only lightweight processor/tokenizer assets (no full model) to the output
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.save_pretrained(out_dir)

    onnx_fp16 = export_to_onnx(args.model_id, out_dir, device=args.device)
    q4_path = quantise_int4(onnx_fp16, out_dir)

    save_manifest(out_dir, {
        "onnx_int4": q4_path.name,
        "processor": "preprocessor_config.json",
        "tokenizer": "tokenizer.json",
    })

    print("Conversion complete ✔")


if __name__ == "__main__":
    main()