import React, { useState } from 'react';
import { TraceStep } from '../../hooks/useAgentChat';
import { ChevronDown, ChevronRight, Terminal, Brain, Eye } from 'lucide-react';
import clsx from 'clsx';

interface ReActTraceViewerProps {
  traces: TraceStep[];
}

export function ReActTraceViewer({ traces }: ReActTraceViewerProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!traces || traces.length === 0) return null;

  return (
    <div className="mt-3 bg-black/30 rounded border border-slate-700 overflow-hidden text-sm">
      <button 
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-2 hover:bg-slate-800/50 transition-colors text-slate-400 font-mono text-xs"
      >
        <span className="flex items-center gap-2">
          {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          Agent Reasoning Trace ({traces.length} steps)
        </span>
      </button>
      
      {isExpanded && (
        <div className="p-2 space-y-2 border-t border-slate-700 max-h-64 overflow-y-auto custom-scrollbar">
          {traces.map((trace, idx) => (
            <div key={idx} className="flex gap-2 items-start">
              <div className="mt-0.5">
                {trace.type === 'Thought' && <Brain size={14} className="text-purple-400" />}
                {trace.type === 'Action' && <Terminal size={14} className="text-amber-400" />}
                {trace.type === 'Observation' && <Eye size={14} className="text-emerald-400" />}
              </div>
              <div className="flex-1">
                <span className={clsx(
                  "font-bold mr-2 text-xs uppercase tracking-wider",
                  trace.type === 'Thought' ? "text-purple-400" :
                  trace.type === 'Action' ? "text-amber-400" : "text-emerald-400"
                )}>
                  {trace.type}:
                </span>
                <span className="font-mono text-slate-300 text-xs break-words">
                  {trace.content}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
