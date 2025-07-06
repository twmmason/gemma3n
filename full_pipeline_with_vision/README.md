# Gemma-3n 4B Vision-Language Browser Demo

This project provides a complete pipeline for converting Google's Gemma-3n 4B vision-language model to INT4 quantized ONNX format and running it in the browser using ONNX-Runtime-Web with WebGPU acceleration.

## Features

- **Complete Model Conversion**: Downloads and converts Gemma-3n 4B vision-language model to INT4 ONNX
- **SigLIP Vision Encoder**: Includes patched vision encoder that works around torch.export issues
- **Browser-Ready**: Optimized for in-browser inference with WebGPU acceleration
- **INT4 Quantization**: Uses AWQ (Activation-aware Weight Quantization) for optimal compression
- **Web Demo**: Ready-to-use HTML/JS interface for testing

## Quick Start

### 1. Install Dependencies

```bash
# Install requirements
pip install -r requirements_venv.txt

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements_venv.txt
```

### 2. Convert Model

```bash
# Convert Gemma-3n 4B to INT4 ONNX (requires ~37GB download + processing time)
python convert_gemma3n_vision_int4.py \
    --model-id "google/gemma-3n-vision-4b-it" \
    --output-dir ./model_int4
```

**Note**: This process will:
- Download ~37GB model files from Hugging Face
- Apply SigLIP vision encoder patches
- Export to ONNX format with FP16 precision
- Quantize to INT4 using AWQ algorithm
- Generate `model_info.json` manifest

### 3. Serve Web Demo

```bash
# Serve the demo (requires a local web server due to CORS)
python -m http.server 8000

# Open browser to: http://localhost:8000/web_demo/
```

### 4. Test the Demo

1. Upload an image (PNG/JPEG)
2. Enter an optional text prompt
3. Click "Run Inference"
4. Wait for model loading (first time only)
5. View generated text description

## File Structure

```
gemma3_browser_vision_demo/
├── convert_gemma3n_vision_int4.py  # Main conversion script
├── requirements_venv.txt           # Python dependencies
├── Dockerfile                      # Optional containerized build
├── web_demo/
│   ├── index.html                  # Web interface
│   └── app.js                      # JavaScript inference code
└── model_int4/                     # Generated model files (after conversion)
    ├── gemma3n_vision_fp16.onnx.q4 # INT4 quantized model
    ├── tokenizer.json              # Tokenizer configuration
    └── model_info.json             # Model metadata
```

## Technical Details

### Model Conversion Pipeline

1. **Download**: Fetches Gemma-3n 4B vision-language model from Hugging Face
2. **Patch**: Applies SigLIP vision encoder patches to fix torch.export issues
3. **Export**: Converts to ONNX format using Optimum exporters
4. **Quantize**: Applies INT4 AWQ quantization for browser optimization
5. **Package**: Creates browser-ready model bundle

### Browser Inference

- **Runtime**: ONNX-Runtime-Web with WebGPU backend
- **Tokenization**: Transformers.js for text processing
- **Image Processing**: SigLIP image processor for vision inputs
- **Quantization**: INT4 weights for reduced memory usage

### Performance Considerations

- **Model Size**: ~2-3GB after INT4 quantization (vs ~15GB original)
- **Memory**: Requires WebGPU-capable browser and sufficient GPU memory
- **Speed**: Inference time depends on image size and prompt length
- **Compatibility**: Modern browsers with WebGPU support

## Troubleshooting

### Import Errors
If you encounter Optimum import errors, ensure you're using compatible versions:
```bash
pip install optimum==1.21.1 torch==2.1.2+cpu numpy<1.25
```

### WebGPU Issues
- Ensure your browser supports WebGPU
- Check GPU memory availability
- Try reducing image resolution

### Model Loading Failures
- Verify model files exist in `model_int4/` directory
- Check network connectivity for CDN resources
- Ensure web server is serving files correctly

## Requirements

### Python Dependencies
- torch==2.1.2+cpu
- transformers>=4.40.0
- optimum==1.21.1
- numpy<1.25
- onnxruntime

### Browser Requirements
- WebGPU support (Chrome 113+, Edge 113+)
- Modern JavaScript (ES6 modules)
- Sufficient GPU memory (4GB+ recommended)

## License

This project follows the same license terms as the underlying Gemma model. Please refer to Google's Gemma license for usage terms.

## Contributing

Contributions welcome! Areas for improvement:
- Performance optimizations
- Additional quantization methods
- Better error handling
- Mobile browser support
- Alternative model formats

## Author

Created by Roo (2025-07-04) as a complete reference implementation for browser-based vision-language model deployment.