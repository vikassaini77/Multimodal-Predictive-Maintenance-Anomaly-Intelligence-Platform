import React, { useEffect, useState, useRef } from 'react';
import clsx from 'clsx';

interface StreamingMessageProps {
  content: string;
  isStreaming: boolean;
}

export function StreamingMessage({ content, isStreaming }: StreamingMessageProps) {
  return (
    <div className="relative">
      <div className="text-sm text-slate-200 whitespace-pre-wrap leading-relaxed">
        {content}
        {isStreaming && (
          <span className="inline-block w-2 h-4 ml-1 align-middle bg-primary animate-pulse" />
        )}
      </div>
    </div>
  );
}
