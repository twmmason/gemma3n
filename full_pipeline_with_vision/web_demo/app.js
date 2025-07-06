// app.js
// Minimal vision-language inference using ONNX-Runtime-Web + Transformers.js
// Serves as reference PoC – performance/debug improvements welcome.

import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.15.0/dist/transformers.esm.min.js';

const resultDiv = document.getElementById('result');
const runBtn = document.getElementById('runBtn');
const imgInput = document.getElementById('imageInput');
const promptEl = document.getElementById('prompt');

// Lazily initialised global pipeline
let gemmaPipe = null;

async function ensurePipeline() {
  if (gemmaPipe) return gemmaPipe;

  // Tell transformers.js to use ORT-WebGPU backend
  globalThis.onnxruntime = ort;                 // expose ORT for transformers.js
  ort.env.wasm.numThreads = 1;                  // small perf tweak
  ort.env.wasm.simd = true;
  ort.env.logLevel = 'warning';

  resultDiv.textContent = 'Loading INT4 model (first time only)…';

  // The model folder is served one level up from web_demo/
  gemmaPipe = await pipeline(
    'image-to-text',
    '../model_int4',          // path resolved by fetch relative to index.html
    {
      quantized: true,        // let ts.js pick *.onnx.q4
      // 'image_processor': 'SiglipImageProcessorFast' – auto-detected
      progress_callback: (p) => {
        resultDiv.textContent = `Downloading: ${(p * 100).toFixed(1)} %`;
      },
    },
  );

  return gemmaPipe;
}

async function runInference() {
  try {
    const file = imgInput.files[0];
    if (!file) {
      alert('Select an image first');
      return;
    }

    const prompt = promptEl.value || '';
    const reader = new FileReader();
    reader.onload = async (e) => {
      const imgData = e.target.result;

      const pipe = await ensurePipeline();
      resultDiv.textContent = 'Running…';

      const { generated_text } = await pipe(imgData, { prompt });
      resultDiv.textContent = generated_text;
    };
    reader.readAsDataURL(file);
  } catch (err) {
    console.error(err);
    resultDiv.textContent = `Error: ${err.message}`;
  }
}

runBtn.addEventListener('click', runInference);