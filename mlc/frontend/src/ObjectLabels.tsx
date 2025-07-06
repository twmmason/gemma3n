import React, { useState } from 'react';
import { useGemma3n } from './useGemma3n';

export default function ObjectLabels() {
  const { ready, send, log } = useGemma3n();
  const [imgUrl, setUrl] = useState<string>();

  const ask = (file: File) => {
    const url = URL.createObjectURL(file);
    setUrl(url);
    const prompt = `![image](${url})\nList unique objects as JSON array.`;
    send(prompt);
  };

  return (
    <div className="space-y-3 max-w-xl mx-auto">
      <input
        type="file"
        accept="image/*"
        disabled={!ready}
        onChange={e => e.target.files && ask(e.target.files[0])}
      />
      {imgUrl && <img src={imgUrl} alt="upload" className="max-h-60" />}
      <pre className="bg-gray-100 p-2 whitespace-pre-wrap">
        {log.slice(-1)}
      </pre>
    </div>
  );
}