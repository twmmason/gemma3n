import { useEffect, useRef, useState } from 'react';

export function useGemma3n() {
  const worker = useRef<Worker>();
  const [ready, setReady] = useState(false);
  const [log, setLog] = useState<string[]>([]);

  useEffect(() => {
    worker.current = new Worker(
      new URL('./gemma3n.worker.ts', import.meta.url),
      { type: 'module' }
    );
    worker.current.onmessage = e => {
      if (e.data.ready) setReady(true);
      else if (e.data.content) setLog(l => [...l, e.data.content]);
    };
    return () => worker.current?.terminate();
  }, []);

  const send = (prompt: string) => {
    worker.current!.postMessage({ prompt });
  };

  return { ready, log, send };
}