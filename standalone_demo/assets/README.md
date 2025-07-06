# Model Files Directory

This directory should contain the Gemma 3n model file required for the demo.

## Required Model File

**File**: `gemma-3n-E2B-it-int4.task`
**Size**: ~3.1GB
**Format**: MediaPipe LiteRT task file

## How to Obtain the Model

You need to download the Gemma 3n E2B INT4 model file. You can:

1. **Copy from existing installation** (if you have it):
   ```bash
   cp /path/to/existing/gemma-3n-E2B-it-int4.task ./assets/
   ```

2. **Download from Hugging Face**:
   - Repository: `google/gemma-3n-E2B-it-litert-lm-preview`
   - File: `gemma-3n-E2B-it-int4.litertlm`
   - Rename to: `gemma-3n-E2B-it-int4.task`

3. **Use the download script** (if available in your original setup):
   ```bash
   python download_correct_model.py
   ```

## Verification

Once downloaded, verify the file:
```bash
ls -la assets/gemma-3n-E2B-it-int4.task
```

The file should be approximately 3.1GB in size.

## File Structure
```
assets/
└── gemma-3n-E2B-it-int4.task  (3.1GB - you need to download this)
```

**Important**: The demo will not work without this model file in place.