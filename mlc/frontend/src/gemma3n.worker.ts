import { LLMWorkerHandler } from '@mlc-ai/web-llm';

const handler = new LLMWorkerHandler();

// relay messages from main thread to WebLLM handler
self.onmessage = evt => handler.postMessage(evt);

// load the Gemma model; notify UI when ready
handler
  .loadModel('/gemma3n/gemma3n-webgpu.json', {
    modelId: 'gemma3n',
    gpuMemoryFraction: 0.9 // tune per device
  })
  .then(() => postMessage({ ready: true }));