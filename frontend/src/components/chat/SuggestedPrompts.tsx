import React from 'react';
import { Sparkles } from 'lucide-react';

interface SuggestedPromptsProps {
  onSelect: (prompt: string) => void;
}

export function SuggestedPrompts({ onSelect }: SuggestedPromptsProps) {
  const prompts = [
    "Why is Machine 4 flagged?",
    "Show bearing fault history for CNC-01",
    "Compare acoustic anomalies across all mills",
    "What is the recommended maintenance action for STAMP-01?"
  ];

  return (
    <div className="flex flex-wrap gap-2 mt-2">
      {prompts.map((prompt, idx) => (
        <button
          key={idx}
          onClick={() => onSelect(prompt)}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800/80 hover:bg-primary/20 border border-slate-700 hover:border-primary/50 text-slate-300 hover:text-primary text-xs rounded-full transition-colors whitespace-nowrap"
        >
          <Sparkles size={12} className="opacity-70" />
          {prompt}
        </button>
      ))}
    </div>
  );
}
