# 🚀 Gemma3n Vision Model Conversion - Colab Notebook

This repository contains an enhanced Google Colab notebook for converting Gemma3n vision-language models to ONNX format with comprehensive verbose logging and time estimation.

## 📋 Features

- **🔍 Enhanced Verbose Logging**: Real-time timestamps, memory monitoring, and detailed progress tracking
- **⏰ Time Estimation**: Intelligent estimation for each conversion phase
- **📊 Resource Monitoring**: CPU, RAM, and GPU usage tracking
- **📁 File Management**: Automatic output organization and manifest generation
- **🎮 Colab Optimized**: Designed specifically for Google Colab environments

## 🚀 Quick Start

### Option 1: Direct Link
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/gemma3n-browser-vision/blob/main/gemma3n_conversion_colab.ipynb)

### Option 2: Manual Upload
1. Download `gemma3n_conversion_colab.ipynb`
2. Upload to Google Colab
3. Run all cells sequentially

## 📦 What the Notebook Does

### Phase 1: Environment Setup
- Checks runtime specifications (CPU, RAM, GPU)
- Installs required dependencies
- Loads the latest transformers library

### Phase 2: Model Inspection
- Loads Gemma3n model for analysis
- Counts model parameters
- Checks architecture components
- Saves tokenizer/processor

### Phase 3: Results Generation
- Creates detailed manifest file
- Displays resource usage statistics
- Provides download functionality

## 🎯 Supported Models

| Model ID | Effective Params | RAM Required | Notes |
|----------|------------------|--------------|-------|
| `google/gemma-3n-E2B-it` | ~2B | 8-12 GB | ✅ Colab Free |
| `google/gemma-3n-E4B-it` | ~4B | 16-24 GB | ⚠️ Colab Pro |
| `google/gemma-3-4b-it` | ~4B | 16-24 GB | ⚠️ Colab Pro |

## 💻 Runtime Requirements

### Minimum (Model Inspection Only)
- **RAM**: 4-8 GB
- **GPU**: Optional (speeds up loading)
- **Storage**: 2-5 GB

### Recommended (Full Conversion)
- **RAM**: 16-32 GB
- **GPU**: 12+ GB VRAM
- **Storage**: 10-20 GB

## 📊 Sample Output

```
[12:34:56] [PHASE] [0.0s] [RAM: 1.2GB] 🚀 Starting phase: Model Loading and Inspection (estimated: 2 min)
[12:34:57] [PROGRESS] [1.2s] [RAM: 1.4GB]    Loading model configuration...
[12:35:45] [PROGRESS] [49.3s] [RAM: 8.7GB]    Model loaded with 5,980,000,000 parameters
[12:35:46] [PROGRESS] [50.1s] [RAM: 8.7GB]    Model type: Gemma3nForConditionalGeneration
[12:35:46] [PROGRESS] [50.2s] [RAM: 8.7GB]    ✓ Vision tower found
[12:35:46] [PHASE] [50.2s] [RAM: 8.7GB] ✓ Completed phase 'Model Loading and Inspection' in 0.8 minutes
```

## 🔧 Customization

### Changing the Model
```python
# Edit this line in the test cell
model_id = "google/gemma-3n-E2B-it"  # Change to your preferred model
```

### Adjusting Output Directory
```python
# Edit this line in the test cell
output_dir = "/content/gemma3n_output"  # Change to your preferred path
```

### Adding Full ONNX Conversion
For advanced users with sufficient resources, you can extend the notebook by adding the full ONNX conversion code from `convert_gemma3n_vision_int4_verbose.py`.

## 📋 Generated Files

After running the notebook, you'll get:

```
📁 gemma3n_output/
├── 📄 model_manifest.json      # Model metadata and conversion info
├── 📄 preprocessor_config.json # Image preprocessing configuration
├── 📄 tokenizer.json          # Tokenizer configuration
├── 📄 tokenizer_config.json   # Tokenizer settings
└── 📄 special_tokens_map.json # Special token mappings
```

## ⚠️ Important Notes

### Memory Limitations
- **Colab Free**: Limited to ~12 GB RAM, suitable for model inspection only
- **Colab Pro**: ~25 GB RAM, can handle smaller models (2B-4B)
- **Full Conversion**: Requires 32+ GB RAM for larger models

### Time Estimates
- **Model Inspection**: 2-5 minutes
- **Full ONNX Export**: 30 minutes - 3 hours (depending on model size)
- **INT4 Quantization**: 15-30 minutes additional

### Storage Considerations
- Model weights: 5-50 GB (depending on model size)
- ONNX files: 2-3x larger than original weights
- Quantized files: ~25% of original ONNX size

## 🛠️ Troubleshooting

### Common Issues

**1. Out of Memory Error**
```
RuntimeError: CUDA out of memory
```
- Solution: Use a smaller model or enable CPU-only mode

**2. Model Not Found**
```
OSError: google/gemma-3n-E4B-it does not appear to be a model identifier
```
- Solution: Check model ID spelling and ensure you have access permissions

**3. Transformers Version Error**
```
ValueError: The checkpoint you are trying to load has model type `gemma3n` but Transformers does not recognize this architecture
```
- Solution: Ensure the latest transformers version is installed (handled automatically in the notebook)

### Performance Tips
- Use GPU runtime for faster model loading
- Clear outputs regularly to save memory
- Close unnecessary browser tabs during conversion

## 📝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Share your conversion results

## 📄 License

This project follows the Gemma license terms. Please review Google's Gemma usage policies before using the models.

---

**Happy converting!** 🎉

For more advanced conversion options, check out the full `convert_gemma3n_vision_int4_verbose.py` script in the repository. 