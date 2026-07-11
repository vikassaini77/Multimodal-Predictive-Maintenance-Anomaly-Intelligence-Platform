import React from 'react';
import { Camera } from 'lucide-react';

interface GradCamViewerProps {
  imageUrl?: string;
}

export function GradCamViewer({ imageUrl }: GradCamViewerProps) {
  if (!imageUrl) return null;

  return (
    <div className="mt-3 bg-slate-800 rounded border border-slate-700 overflow-hidden">
      <div className="bg-slate-900 px-3 py-1.5 flex items-center gap-2 border-b border-slate-700">
        <Camera size={14} className="text-primary" />
        <span className="text-xs font-mono text-slate-300">GradCAM Analysis</span>
      </div>
      <div className="p-2 flex justify-center bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] bg-slate-800">
        <img 
          src={imageUrl} 
          alt="GradCAM Heatmap" 
          className="max-h-48 rounded shadow-lg border border-slate-700/50"
        />
      </div>
    </div>
  );
}
