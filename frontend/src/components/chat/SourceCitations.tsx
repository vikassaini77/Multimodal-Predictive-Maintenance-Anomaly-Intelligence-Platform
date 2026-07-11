import React from 'react';
import { Citation } from '../../hooks/useAgentChat';
import { BookOpen } from 'lucide-react';

interface SourceCitationsProps {
  citations: Citation[];
}

export function SourceCitations({ citations }: SourceCitationsProps) {
  if (!citations || citations.length === 0) return null;

  return (
    <div className="mt-3 pt-3 border-t border-slate-700/50">
      <div className="flex items-center gap-1.5 text-slate-400 text-xs font-semibold mb-2 uppercase tracking-wider">
        <BookOpen size={12} />
        Sources
      </div>
      <div className="flex flex-wrap gap-2">
        {citations.map((cite, idx) => (
          <div 
            key={idx}
            className="group relative bg-slate-800 border border-slate-700 hover:border-primary/50 text-slate-300 text-xs px-2 py-1 rounded cursor-help transition-colors"
          >
            {cite.document} [{cite.chunk_id}]
            
            {/* Tooltip */}
            <div className="absolute bottom-full left-0 mb-2 w-64 p-2 bg-slate-900 border border-slate-600 rounded shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10 pointer-events-none">
              <p className="font-mono text-[10px] text-slate-400 mb-1">{cite.document}</p>
              <p className="text-xs text-slate-200 line-clamp-4">{cite.text}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
