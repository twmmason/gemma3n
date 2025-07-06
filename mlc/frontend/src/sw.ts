import { precacheAndRoute } from 'workbox-precaching';

// Vite injects the hashed assets list into __WB_MANIFEST
precacheAndRoute(self.__WB_MANIFEST || []);

// Manually list Gemma model shards for offline caching
self.addEventListener('install', (e: ExtendableEvent) => {
  e.waitUntil(
    caches.open('gemma3n').then(c =>
      c.addAll([
        '/gemma3n/gemma3n-webgpu.json',
        '/gemma3n/gemma3n-wasm32.wasm',
        '/gemma3n/params_shard_0.bin',
        '/gemma3n/params_shard_1.bin',
        // … add more shards if generated
      ])
    )
  );
});