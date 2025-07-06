import React, { useState } from 'react';
import { useGemma3n } from './useGemma3n';

export default function GemmaChat() {
  const { ready, log, send } = useGemma3n();
  const [input, setInput] = useState('');

  return (
    <div className="space-y-3 max-w-xl mx-auto">
      <div className="h-72 overflow-y-auto border p-2 rounded">
        {log.map((l, i) => (
          // eslint-disable-next-line react/no-array-index-key
          <p key={i}>{l}</p>
        ))}
      </div>
      <input
        disabled={!ready}
        value={input}
        onChange={e => setInput(e.target.value)}
        onKeyDown={e => {
          if (e.key === 'Enter' && input.trim()) {
            send(input);
            setInput('');
          }
        }}
        className="border p-2 w-full"
        placeholder={ready ? 'Ask Gemma 3n…' : 'Loading model…'}
      />
    </div>
  );
}