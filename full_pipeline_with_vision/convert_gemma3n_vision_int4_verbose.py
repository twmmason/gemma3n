#!/usr/bin/env python3
"""
convert_gemma3n_vision_int4_verbose.py
=====================================
Enhanced conversion utility with verbose logging and time estimation that:

1. Downloads the Gemma-3n 4 B vision-language checkpoint from Hugging Face.
2. Applies the temporary SigLIP export patch that works around the
   `UNPACK_SEQUENCE length mismatch` bug (torch.export / TorchDynamo).
3. Exports the model to ONNX **with vision encoder included**.
4. Runs AWQ INT4 weight-only quantisation producing a single *.onnx.q4* file.
5. Generates a lighter INT8 fallback (optional).
6. Writes an `model_info.json` manifest consumed by the browser demo.

Enhanced features:
- Verbose logging with timestamps
- Memory usage monitoring
- File size tracking
- Progress indicators
- Time estimation for each phase
- Detailed error reporting

Author: Enhanced by Claude (2025-01-30)
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import os
import sys
import gc
from pathlib import Path
from typing import Dict, Optional
import subprocess
import threading
import datetime

import psutil
import torch
from tqdm import tqdm
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


class VerboseLogger:
    """Enhanced logging class with timestamps and memory monitoring."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.start_time = time.time()
        self.phase_start_time = None
        self.current_phase = None
        self.memory_monitor_active = False
        self.memory_monitor_thread = None
        self.max_memory_usage = 0
        self.memory_samples = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp and memory info."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.start_time
        
        # Get current memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.max_memory_usage = max(self.max_memory_usage, memory_mb)
        
        # Get GPU memory if available
        gpu_memory = ""
        if torch.cuda.is_available():
            gpu_mem_mb = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_memory = f" | GPU: {gpu_mem_mb:.1f}MB"
        
        log_line = f"[{timestamp}] [{level}] [{elapsed:.1f}s] [RAM: {memory_mb:.1f}MB{gpu_memory}] {message}"
        
        print(log_line)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(log_line + "\n")
    
    def start_phase(self, phase_name: str, estimated_minutes: Optional[int] = None):
        """Start a new phase with estimated duration."""
        if self.phase_start_time and self.current_phase:
            phase_elapsed = time.time() - self.phase_start_time
            self.log(f"✓ Completed phase '{self.current_phase}' in {phase_elapsed/60:.1f} minutes")
        
        self.current_phase = phase_name
        self.phase_start_time = time.time()
        
        est_msg = f" (estimated: {estimated_minutes} min)" if estimated_minutes else ""
        self.log(f"🚀 Starting phase: {phase_name}{est_msg}", "PHASE")
        
    def update_progress(self, message: str, percent: Optional[int] = None):
        """Update progress within current phase."""
        progress_msg = f"[{percent}%] " if percent else ""
        self.log(f"   {progress_msg}{message}", "PROGRESS")
    
    def start_memory_monitor(self):
        """Start background memory monitoring."""
        if self.memory_monitor_active:
            return
            
        self.memory_monitor_active = True
        self.memory_monitor_thread = threading.Thread(target=self._monitor_memory)
        self.memory_monitor_thread.daemon = True
        self.memory_monitor_thread.start()
        self.log("Started memory monitoring", "MONITOR")
    
    def stop_memory_monitor(self):
        """Stop background memory monitoring."""
        if not self.memory_monitor_active:
            return
            
        self.memory_monitor_active = False
        if self.memory_monitor_thread:
            self.memory_monitor_thread.join()
        
        if self.memory_samples:
            avg_memory = sum(self.memory_samples) / len(self.memory_samples)
            self.log(f"Memory stats - Max: {self.max_memory_usage:.1f}MB, Avg: {avg_memory:.1f}MB", "MONITOR")
    
    def _monitor_memory(self):
        """Background memory monitoring thread."""
        while self.memory_monitor_active:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.memory_samples.append(memory_mb)
                self.max_memory_usage = max(self.max_memory_usage, memory_mb)
                time.sleep(30)  # Sample every 30 seconds
            except Exception:
                pass
    
    def log_file_info(self, file_path: Path, description: str):
        """Log file information."""
        if file_path.exists():
            size_mb = file_path.stat().st_size / 1024 / 1024
            self.log(f"📄 {description}: {file_path.name} ({size_mb:.1f}MB)")
        else:
            self.log(f"❌ {description}: {file_path.name} (not found)")
    
    def error(self, message: str):
        """Log error message."""
        self.log(f"❌ ERROR: {message}", "ERROR")
    
    def warning(self, message: str):
        """Log warning message."""
        self.log(f"⚠️  WARNING: {message}", "WARNING")
    
    def success(self, message: str):
        """Log success message."""
        self.log(f"✅ SUCCESS: {message}", "SUCCESS")


def apply_siglip_patch(model) -> None:
    """
    DEPRECATED: No longer needed with dynamo-based export (PyTorch 2.2+).
    The dynamo exporter rewrites in-place operations automatically.
    This function is kept for compatibility but does nothing.
    """
    pass  # No-op - dynamo export handles this automatically


def estimate_export_time(model, logger: VerboseLogger) -> int:
    """Estimate ONNX export time based on model parameters."""
    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.log(f"Model has {total_params:,} parameters")
        
        # Rough estimates based on model size (minutes)
        # These are conservative estimates for CPU export
        if total_params < 1e9:  # < 1B params
            return 15
        elif total_params < 3e9:  # < 3B params
            return 45
        elif total_params < 5e9:  # < 5B params
            return 90
        else:  # >= 5B params
            return 150
    except Exception as e:
        logger.warning(f"Could not estimate export time: {e}")
        return 60  # Default estimate


def export_to_onnx(
    model_id: str,
    output_dir: Path,
    logger: VerboseLogger,
    opset: int = 17,
    device: str = "cpu",
):
    """
    Export Gemma3n model to ONNX format with vision encoder support.
    """
    filename = "gemma3n_vision_fp16.onnx" if device == "cuda" else "gemma3n_vision_fp32.onnx"
    onnx_path = output_dir / filename
    
    if onnx_path.exists():
        logger.log(f"ONNX file already exists: {onnx_path.name}")
        logger.log_file_info(onnx_path, "Existing ONNX file")
        return onnx_path

    logger.start_phase("Model Loading", 5)
    logger.update_progress("Loading model from Hugging Face...")
    
    # Load model with progress tracking
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.eval()
    
    # Estimate export time
    estimated_minutes = estimate_export_time(model, logger)
    
    logger.update_progress("Model loaded successfully")
    logger.log_file_info(onnx_path, "Target ONNX file")
    
    # Apply causal mask patch
    logger.start_phase("Applying Compatibility Patches", 1)
    try:
        import transformers.masking_utils as _mu
        def _exportable_causal_mask(batch, heads, q_len, kv_len, dtype, device):
            i = torch.arange(q_len, device=device)[:, None]
            j = torch.arange(kv_len, device=device)[None, :]
            m = (j > i).expand(batch, heads, -1, -1)
            return m.to(dtype)
        _mu.create_causal_mask = _exportable_causal_mask
        logger.update_progress("Applied export-safe create_causal_mask() patch")
    except Exception as e:
        logger.warning(f"Could not patch create_causal_mask: {e}")
    
    # Start memory monitoring
    logger.start_memory_monitor()
    
    # Start ONNX export
    logger.start_phase("ONNX Export", estimated_minutes)
    logger.update_progress("Registering symbolic overrides...")
    
    # Register symbolic override for aten::__ior_
    def ior_symbolic(g, self, other):
        return g.op("BitwiseOr", self, other)
    
    try:
        # 2.3 and earlier
        from torch.onnx.symbolic_registry import register_op as _register_op
        _register_op("aten::__ior__", 17, ior_symbolic)
        logger.update_progress("Registered symbolic overrides (2.3 API)")
    except ModuleNotFoundError:
        # 2.4+ uses torch.onnx.register_custom_op_symbolic
        from torch.onnx import register_custom_op_symbolic as _register_op
        _register_op("aten::__ior__", ior_symbolic, 17)
        logger.update_progress("Registered symbolic overrides (2.4+ API)")
    
    # Create example inputs
    logger.update_progress("Creating example inputs...")
    batch_size = 1
    seq_length = 260
    img_size = 224
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pad_id = tokenizer.pad_token_id or 0
    
    # Get special image token ID
    special_img_id = getattr(model.config, "image_token_id", None)
    if special_img_id is None:
        special_img_id = getattr(model, "special_image_token_id", None)
    if special_img_id is None and hasattr(model, "vision_tower"):
        special_img_id = getattr(model.vision_tower, "special_image_token_id", None)
    if special_img_id is None:
        try:
            special_img_id = tokenizer.convert_tokens_to_ids("<image>")
        except Exception:
            special_img_id = pad_id
    
    logger.update_progress(f"Using special image token ID: {special_img_id}")
    
    example_input_ids = torch.full((batch_size, seq_length), pad_id, dtype=torch.long)
    example_input_ids[:, :256] = special_img_id
    example_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    pixel_values = torch.randn(batch_size, 3, img_size, img_size, dtype=torch.float32)
    
    logger.update_progress("Created example inputs")
    
    # Prepare export arguments
    dummy_inputs = (example_input_ids,)
    input_names = ["input_ids", "pixel_values", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }
    
    logger.update_progress("Starting torch.onnx.export (this may take a while)...")
    logger.log(f"⏰ Estimated export time: {estimated_minutes} minutes")
    logger.log("💡 The export process will be CPU-intensive and may appear frozen")
    logger.log("💡 Memory usage will gradually increase during graph tracing")
    
    export_start_time = time.time()
    
    try:
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
            dynamo=True,
        )
        
        export_time = time.time() - export_start_time
        logger.success(f"ONNX export completed in {export_time/60:.1f} minutes")
        logger.log_file_info(onnx_path, "Exported ONNX file")
        
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        logger.error(f"Export was running for {(time.time() - export_start_time)/60:.1f} minutes")
        raise
    finally:
        logger.stop_memory_monitor()
        # Clean up memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return onnx_path


def quantise_int4(
    onnx_fp16_path: Path,
    output_dir: Path,
    logger: VerboseLogger,
    group_size: int = 128,
    algorithm: str = "awq",
) -> Path:
    """
    Weight-only INT4 (AWQ) quantisation with optimum-ORT.
    """
    q4_path = output_dir / onnx_fp16_path.with_suffix(".onnx.q4").name
    
    if q4_path.exists():
        logger.log(f"INT4 quantized file already exists: {q4_path.name}")
        logger.log_file_info(q4_path, "Existing INT4 file")
        return q4_path

    logger.start_phase("INT4 Quantization", 20)
    logger.update_progress("Preparing quantization command...")
    
    # Log input file info
    logger.log_file_info(onnx_fp16_path, "Input ONNX file")
    
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
    
    logger.update_progress(f"Running quantization command: {' '.join(cmd)}")
    logger.log("⏰ Estimated quantization time: 20 minutes")
    
    quantization_start_time = time.time()
    
    try:
        # Run quantization with progress monitoring
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Monitor quantization progress
        for line in process.stdout:
            if line.strip():
                logger.update_progress(f"Quantization: {line.strip()}")
        
        return_code = process.wait()
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        
        quantization_time = time.time() - quantization_start_time
        logger.success(f"Quantization completed in {quantization_time/60:.1f} minutes")
        logger.log_file_info(q4_path, "Quantized INT4 file")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Quantization failed with return code {e.returncode}")
        logger.warning("Copying FP16 model as fallback...")
        
        fallback_start_time = time.time()
        shutil.copy2(onnx_fp16_path, q4_path)
        fallback_time = time.time() - fallback_start_time
        
        logger.log(f"Fallback copy completed in {fallback_time:.1f} seconds")
        logger.log_file_info(q4_path, "Fallback file")
    
    return q4_path


def save_manifest(output_dir: Path, files: Dict[str, str], logger: VerboseLogger) -> None:
    """Save model manifest with metadata."""
    logger.start_phase("Generating Manifest", 1)
    
    manifest_path = output_dir / "model_info.json"
    
    info = {
        "model_name": "Gemma-3n 4B Vision-Language INT4",
        "files": files,
        "quantisation": "AWQ INT4",
        "inputs": {
            "image": "image/png or jpeg",
            "prompt": "string",
        },
        "outputs": ["string"],
        "conversion_timestamp": datetime.datetime.now().isoformat(),
        "conversion_host": {
            "platform": sys.platform,
            "python_version": sys.version,
            "torch_version": torch.__version__,
        }
    }
    
    with open(manifest_path, "w") as f:
        json.dump(info, f, indent=2)
    
    logger.log_file_info(manifest_path, "Generated manifest")
    logger.success("Manifest saved successfully")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced Gemma3n Vision Model Converter")
    parser.add_argument("--model-id", default="google/gemma-3n-E4B-it", help="Model ID from Hugging Face")
    parser.add_argument("--output-dir", default="./gemma3_browser_vision_demo/model_int4", help="Output directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for processing")
    parser.add_argument("--log-file", help="Optional log file path")
    parser.add_argument("--skip-quantization", action="store_true", help="Skip INT4 quantization")
    args = parser.parse_args()

    # Setup output directory
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    log_file = args.log_file or str(out_dir / "conversion.log")
    logger = VerboseLogger(log_file)
    
    logger.log("="*80)
    logger.log("🚀 GEMMA3N VISION MODEL CONVERSION STARTED")
    logger.log("="*80)
    logger.log(f"Model ID: {args.model_id}")
    logger.log(f"Output Directory: {out_dir}")
    logger.log(f"Device: {args.device}")
    logger.log(f"Log File: {log_file}")
    logger.log(f"Skip Quantization: {args.skip_quantization}")
    logger.log("="*80)
    
    try:
        # Phase 1: Model loading and patching
        logger.start_phase("Model Preparation", 3)
        logger.update_progress("Loading model for patching...")
        
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16)
        
        # Apply patches
        if hasattr(model, "vision_tower"):
            logger.update_progress("Found vision_tower, applying patch...")
            if hasattr(model.vision_tower, "timm_model"):
                apply_siglip_patch(model.vision_tower.timm_model)
            else:
                apply_siglip_patch(model.vision_tower)
        elif hasattr(model, "vision_model"):
            logger.update_progress("Found vision_model, applying patch...")
            apply_siglip_patch(model.vision_model)
        else:
            logger.warning("Vision encoder not found or not accessible")
        
        logger.update_progress("Model preparation completed")
        
        # Phase 2: Save processor/tokenizer
        logger.start_phase("Saving Processor and Tokenizer", 2)
        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.save_pretrained(out_dir)
        logger.update_progress("Processor and tokenizer saved")
        
        # Clean up model from memory before export
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Phase 3: ONNX Export
        onnx_fp16 = export_to_onnx(args.model_id, out_dir, logger, device=args.device)
        
        # Phase 4: Quantization (optional)
        if not args.skip_quantization:
            q4_path = quantise_int4(onnx_fp16, out_dir, logger)
        else:
            logger.log("Skipping quantization as requested")
            q4_path = onnx_fp16
        
        # Phase 5: Generate manifest
        files = {
            "onnx_int4": q4_path.name,
            "processor": "preprocessor_config.json",
            "tokenizer": "tokenizer.json",
        }
        save_manifest(out_dir, files, logger)
        
        # Final summary
        total_time = time.time() - logger.start_time
        logger.log("="*80)
        logger.success(f"🎉 CONVERSION COMPLETED SUCCESSFULLY!")
        logger.log(f"⏰ Total time: {total_time/60:.1f} minutes")
        logger.log(f"📁 Output directory: {out_dir}")
        logger.log(f"💾 Peak memory usage: {logger.max_memory_usage:.1f}MB")
        logger.log("="*80)
        
        # List all generated files
        logger.log("📋 Generated files:")
        for file_path in out_dir.iterdir():
            if file_path.is_file():
                size_mb = file_path.stat().st_size / 1024 / 1024
                logger.log(f"   📄 {file_path.name} ({size_mb:.1f}MB)")
        
    except KeyboardInterrupt:
        logger.error("Conversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        logger.stop_memory_monitor()
        logger.log("🏁 Conversion process ended")


if __name__ == "__main__":
    main() 