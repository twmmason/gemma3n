#!/usr/bin/env python3
"""Debug script to understand the ONNX export issues."""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import traceback

def test_model_export():
    print("Loading model...")
    model_id = "google/gemma-3n-E4B-it"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    
    print("\nModel structure:")
    print(f"Model type: {type(model)}")
    print(f"Has vision_tower: {hasattr(model, 'vision_tower')}")
    print(f"Has embed_vision: {hasattr(model, 'embed_vision')}")
    
    if hasattr(model, 'vision_tower'):
        print(f"Vision tower type: {type(model.vision_tower)}")
        if hasattr(model.vision_tower, 'timm_model'):
            print(f"Timm model type: {type(model.vision_tower.timm_model)}")
    
    # Try different export approaches
    print("\n\n=== Approach 1: Export text-only model ===")
    try:
        class TextOnlyWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, input_ids, attention_mask=None):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.logits
        
        wrapper = TextOnlyWrapper(model)
        wrapper.eval()
        
        # Example inputs
        batch_size = 1
        seq_length = 128
        example_input_ids = torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.long)
        example_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
        
        # Test forward pass
        with torch.no_grad():
            output = wrapper(example_input_ids, example_attention_mask)
            print(f"Output shape: {output.shape}")
        
        # Try export using torch.onnx.export with dynamo=True (PyTorch 2.2+)
        # This avoids UnsupportedOperatorError for in-place ops like aten::__ior_
        torch.onnx.export(
            wrapper,
            (example_input_ids,),
            "test_text_only.onnx",
            kwargs={"attention_mask": example_attention_mask},
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            opset_version=17,
            dynamo=True,  # Enable the new exporter that handles in-place ops
            dynamic_shapes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
            }
        )
        
        print("✓ Text-only export succeeded using dynamo-based export!")
        
        # Check file size
        import os
        size = os.path.getsize("test_text_only.onnx")
        print(f"File size: {size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"✗ Text-only export failed: {e}")
        traceback.print_exc()
    
    print("\n\n=== Approach 2: Export with vision using proper forward ===")
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Create a more realistic example
        from PIL import Image
        import numpy as np
        
        # Create dummy image
        dummy_image = Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
        
        # Process inputs
        inputs = processor(
            text="<image>What is in this image?",
            images=dummy_image,
            return_tensors="pt"
        )
        
        print(f"Processor output keys: {list(inputs.keys())}")
        
        # Test forward pass with real inputs
        with torch.no_grad():
            outputs = model(**inputs)
            print(f"Model output type: {type(outputs)}")
            print(f"Logits shape: {outputs.logits.shape}")
        
    except Exception as e:
        print(f"✗ Vision test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_model_export()