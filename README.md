# Gemma-3 n WebGPU Demo

This repository contains a **fully client-side** WebGPU demo for Google’s **Gemma-3 n E4B-it** model.  
Compiled with **MLC LLM + WebLLM**, the weights run directly in the browser—no servers, no API keys.

## What’s here?

| Path                           | Purpose                                                |
| ------------------------------ | ------------------------------------------------------ |
| `scripts/compile_gemma3n.sh`   | One-shot script to compile + quantise the FP16 weights |
| `frontend/`                    | Vite + React 18 project                                |
| `frontend/public/gemma3n/`     | Model shards produced by the script (empty in git)     |
| `frontend/src/`                | Web Worker, SW, hooks, and React UIs                   |

## Quick start

```bash
# 1. Install deps
npm --prefix frontend ci

# 2. (Once) compile weights – see scripts/compile_gemma3n.sh
#    Requires a CUDA box with 4 GB+ VRAM.

# 3. Launch dev server
npm --prefix frontend run dev
```

Open <http://localhost:5173> and chat or label objects offline.

## Further reading

The full end-to-end instructions live in the **build book** bundled with this repo’s documentation.  
Follow it for deeper details on compilation flags, service-worker caching, and deployment tips.