import React, { useState } from 'react';
import GemmaChat from './GemmaChat';
import ObjectLabels from './ObjectLabels';

export default function App() {
  const [page, setPage] = useState<'chat' | 'vision'>('chat');

  return (
    <main className="p-4 font-sans">
      <nav className="mb-4 space-x-4">
        <button
          type="button"
          onClick={() => setPage('chat')}
          className={page === 'chat' ? 'font-bold underline' : ''}
        >
          Chat
        </button>
        <button
          type="button"
          onClick={() => setPage('vision')}
          className={page === 'vision' ? 'font-bold underline' : ''}
        >
          Object Labels
        </button>
      </nav>

      {page === 'chat' ? <GemmaChat /> : <ObjectLabels />}
    </main>
  );
}