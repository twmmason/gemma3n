#!/usr/bin/env python3
"""
Test the exported ONNX model to verify it can handle vision inputs.
"""

import numpy as np
import onnxruntime as ort
from pathlib import Path
import json

def test_onnx_model():
    # Load model info
    model_dir = Path("./model_int4")
    with open(model_dir / "model_info.json", "r") as f:
        model_info = json.load(f)
    
    model_path = model_dir / model_info["files"]["onnx_model"]
    
    print(f"Loading ONNX model from: {model_path}")
    
    # Create ONNX Runtime session
    session = ort.InferenceSession(str(model_path))
    
    # Get input/output info
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    
    print("\nModel inputs:")
    for inp in inputs:
        print(f"  - {inp.name}: {inp.shape} ({inp.type})")
    
    print("\nModel outputs:")
    for out in outputs:
        print(f"  - {out.name}: {out.shape} ({out.type})")
    
    # Create dummy inputs
    batch_size = 1
    seq_length = 128
    image_size = 384
    
    input_ids = np.random.randint(0, 1000, (batch_size, seq_length), dtype=np.int64)
    pixel_values = np.random.randn(batch_size, 3, image_size, image_size).astype(np.float32)
    attention_mask = np.ones((batch_size, seq_length), dtype=np.int64)
    
    # Run inference
    print("\nRunning inference with dummy inputs...")
    try:
        feed_dict = {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'attention_mask': attention_mask
        }
        
        outputs = session.run(None, feed_dict)
        print(f"✓ Inference successful!")
        print(f"  Output shape: {outputs[0].shape}")
        print(f"  Output dtype: {outputs[0].dtype}")
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_onnx_model()