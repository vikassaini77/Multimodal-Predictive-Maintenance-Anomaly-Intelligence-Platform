import React, { useState, useRef, useEffect } from 'react';
import { useAgentChat } from '../../hooks/useAgentChat';
import { StreamingMessage } from './StreamingMessage';
import { ReActTraceViewer } from './ReActTraceViewer';
import { SourceCitations } from './SourceCitations';
import { GradCamViewer } from './GradCamViewer';
import { SuggestedPrompts } from './SuggestedPrompts';
import { Send, Bot, User, Loader2 } from 'lucide-react';
import clsx from 'clsx';

export function AgentChatPanel() {
  const { messages, sendQuery, isAgentTyping } = useAgentChat('ws://localhost:8000/ws/agent-chat');
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isAgentTyping]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isAgentTyping) return;
    sendQuery(inputValue.trim());
    setInputValue('');
  };

  return (
    <div className="bg-surface rounded-lg p-0 border border-slate-700 shadow-xl flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="bg-slate-800/80 px-4 py-3 border-b border-slate-700 flex justify-between items-center">
        <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <Bot className="text-primary" size={20} />
          Diagnostic Agent
        </h2>
        <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
          <span className="w-2 h-2 rounded-full bg-ok animate-pulse"></span>
          Agent Ready
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6 custom-scrollbar bg-slate-900/50">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center px-4">
            <Bot size={48} className="text-slate-700 mb-4" />
            <h3 className="text-slate-300 font-semibold mb-2">How can I help diagnose?</h3>
            <p className="text-slate-500 text-sm mb-6 max-w-sm">
              I'm an AI engineer assistant equipped with multimodal RAG and predictive models. Ask me anything about the factory floor.
            </p>
            <SuggestedPrompts onSelect={sendQuery} />
          </div>
        ) : (
          messages.map((msg) => (
            <div 
              key={msg.id} 
              className={clsx(
                "flex gap-3 max-w-[90%]",
                msg.role === 'user' ? "ml-auto flex-row-reverse" : "mr-auto"
              )}
            >
              {/* Avatar */}
              <div className="flex-shrink-0 mt-1">
                {msg.role === 'user' ? (
                  <div className="w-8 h-8 rounded-full bg-primary/20 text-primary flex items-center justify-center border border-primary/30">
                    <User size={16} />
                  </div>
                ) : (
                  <div className="w-8 h-8 rounded-full bg-slate-700 text-slate-300 flex items-center justify-center border border-slate-600 shadow-lg">
                    <Bot size={16} />
                  </div>
                )}
              </div>

              {/* Message Bubble */}
              <div className={clsx(
                "rounded-lg p-3 shadow-sm",
                msg.role === 'user' ? "bg-primary text-slate-900 rounded-tr-none" : "bg-slate-800 border border-slate-700 rounded-tl-none w-full"
              )}>
                {msg.role === 'user' ? (
                  <p className="text-sm font-medium">{msg.content}</p>
                ) : (
                  <div className="w-full">
                    <StreamingMessage content={msg.content} isStreaming={!!msg.streaming} />
                    
                    {msg.imageUrl && (
                      <GradCamViewer imageUrl={msg.imageUrl} />
                    )}
                    
                    {msg.traces && msg.traces.length > 0 && (
                      <ReActTraceViewer traces={msg.traces} />
                    )}
                    
                    {msg.citations && msg.citations.length > 0 && (
                      <SourceCitations citations={msg.citations} />
                    )}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        
        {/* Loading Indicator */}
        {isAgentTyping && (
          <div className="flex gap-3 max-w-[90%] mr-auto">
             <div className="flex-shrink-0 mt-1">
                <div className="w-8 h-8 rounded-full bg-slate-700 text-slate-300 flex items-center justify-center border border-slate-600 shadow-lg">
                  <Bot size={16} />
                </div>
              </div>
              <div className="bg-slate-800 border border-slate-700 rounded-lg rounded-tl-none p-3 shadow-sm flex items-center gap-2">
                <Loader2 size={16} className="text-primary animate-spin" />
                <span className="text-xs text-slate-400 font-mono">Agent is thinking...</span>
              </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 bg-slate-800/80 border-t border-slate-700">
        {messages.length > 0 && (
          <div className="mb-3">
             <SuggestedPrompts onSelect={(p) => { if(!isAgentTyping) sendQuery(p); }} />
          </div>
        )}
        <form onSubmit={handleSubmit} className="flex gap-2 relative">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder={isAgentTyping ? "Agent is reasoning..." : "Ask a diagnostic question..."}
            disabled={isAgentTyping}
            className="flex-1 bg-slate-900 border border-slate-600 rounded-full px-4 py-2.5 text-sm text-slate-200 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed placeholder:text-slate-500"
          />
          <button
            type="submit"
            disabled={!inputValue.trim() || isAgentTyping}
            className="bg-primary hover:bg-primary/80 disabled:bg-slate-700 disabled:text-slate-500 text-slate-900 rounded-full w-10 h-10 flex items-center justify-center transition-colors shadow-lg shadow-primary/20 flex-shrink-0"
          >
            <Send size={16} className="ml-0.5" />
          </button>
        </form>
      </div>
    </div>
  );
}
