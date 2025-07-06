# Gemma 3n Standalone Browser Demo

A self-contained demo of Google's Gemma 3n language model running entirely in the browser using MediaPipe and WebGPU acceleration.

## Quick Start

### 1. Requirements
- **Browser**: Chrome 124+, Edge 124+, or Firefox Nightly (with WebGPU enabled)
- **Python**: 3.7+ (for the local server)
- **ngrok**: For HTTPS tunneling (WebGPU requires secure context)

### 2. Get the Model File
Download the Gemma 3n model file (~3.1GB) and place it in the `assets/` directory:

```bash
# Copy from existing installation (if you have it)
cp /path/to/gemma-3n-E2B-it-int4.task ./assets/

# OR download from Hugging Face
# Repository: google/gemma-3n-E2B-it-litert-lm-preview
# File: gemma-3n-E2B-it-int4.litertlm (rename to .task)
```

See `assets/README.md` for detailed download instructions.

### 3. Run the Demo

**Step 1**: Start the local server
```bash
python serve.py
```

**Step 2**: Expose via ngrok (for HTTPS)
```bash
ngrok http --url=gemma3n.ngrok.app 8080
```

**Step 3**: Access the demo
Open: `https://gemma3n.ngrok.app/demo.html`

## What's Included

```
gemma3n-standalone-demo/
├── demo.html          # Complete MediaPipe WebGPU demo interface
├── serve.py           # HTTP server with WebGPU headers
├── assets/            # Model files directory
│   ├── README.md      # Model download instructions
│   └── gemma-3n-E2B-it-int4.task  # Model file (you need to download)
└── README.md          # This file
```

## Demo Features

- **Real-time Generation**: Streaming responses with performance metrics
- **Configurable Parameters**: Max tokens, temperature, top-K sampling
- **WebGPU Acceleration**: Hardware-accelerated inference
- **Browser-based**: No server-side processing required

## Expected Performance

- **First Load**: 10-20 seconds (model initialization)
- **Generation Speed**: ~15-25 tokens/second (varies by GPU)
- **Memory**: ~3-4GB GPU memory required

## Browser Support

| Browser | Version | WebGPU | Status |
|---------|---------|---------|---------|
| Chrome/Edge | 124+ | ✅ Default | Ready |
| Firefox | Nightly 141+ | ⚠️ Enable flag | `dom.webgpu.enabled = true` |
| Safari | Any | ❌ Not supported | Not available |

## Troubleshooting

### Common Issues

1. **"WebGPU not supported"**
   - Use Chrome 124+ or Edge 124+
   - For Firefox: Enable `dom.webgpu.enabled` in `about:config`

2. **"Model file not found"**
   - Ensure `gemma-3n-E2B-it-int4.task` is in `assets/` directory
   - Check file size is ~3.1GB

3. **"Secure context required"**
   - Use ngrok for HTTPS tunneling
   - Or serve from `localhost` (limited WebGPU support)

4. **Out of memory errors**
   - Close other GPU-intensive applications
   - Ensure 4GB+ GPU memory available
   - Reduce max tokens parameter

### Verification Steps

1. Check model file exists: `ls -la assets/gemma-3n-E2B-it-int4.task`
2. Verify server is running: `http://localhost:8080/demo.html`
3. Test ngrok tunnel: `https://gemma3n.ngrok.app/demo.html`
4. Open browser DevTools → Console for WebGPU logs

## Technical Details

- **Model**: Gemma 3n E2B INT4 quantized (2.97GB)
- **Framework**: MediaPipe GenAI with WebGPU backend
- **Acceleration**: GPU-accelerated inference via WebGPU
- **Streaming**: Real-time token generation with callbacks

## License

This demo uses Google's Gemma 3n model. Please refer to the original model license and terms of use.