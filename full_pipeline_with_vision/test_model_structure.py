#!/usr/bin/env python3
"""Test script to explore Gemma3n model structure"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# Load model and explore structure
model_id = "google/gemma-3n-E4B-it"
print(f"Loading model: {model_id}")

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained(model_id)

print("\n=== Model Type ===")
print(f"Model type: {type(model)}")
print(f"Model config type: {model.config.model_type}")

print("\n=== Model Attributes ===")
attributes = [attr for attr in dir(model) if not attr.startswith('_')]
for attr in sorted(attributes):
    if hasattr(model, attr) and not callable(getattr(model, attr)):
        try:
            val = getattr(model, attr)
            print(f"{attr}: {type(val)}")
        except:
            pass

print("\n=== Vision-related Attributes ===")
vision_attrs = [attr for attr in attributes if 'vision' in attr.lower() or 'image' in attr.lower()]
for attr in vision_attrs:
    print(f"  {attr}")

print("\n=== Model Config ===")
print(f"Model config keys: {list(model.config.to_dict().keys())}")

# Check if there's a vision config
if hasattr(model.config, 'vision_config'):
    print("\n=== Vision Config ===")
    print(model.config.vision_config)

# Check for image/vision processing capabilities
print("\n=== Processor Info ===")
print(f"Processor type: {type(processor)}")
print(f"Processor attributes: {[attr for attr in dir(processor) if not attr.startswith('_') and not callable(getattr(processor, attr))]}")

# Try to find vision encoder
print("\n=== Looking for Vision Encoder ===")
for name, module in model.named_modules():
    if 'vision' in name.lower() or 'image' in name.lower() or 'mobilenet' in name.lower():
        print(f"Found: {name} -> {type(module)}")