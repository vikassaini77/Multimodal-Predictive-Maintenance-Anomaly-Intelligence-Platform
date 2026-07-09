import React from 'react';
import clsx from 'clsx';
import { MachineStats } from '../hooks/useAnomalyFeed';

interface FactoryMapProps {
  machines: Record<string, MachineStats>;
  selectedMachineId: string | null;
  onSelectMachine: (id: string) => void;
}

export function FactoryMap({ machines, selectedMachineId, onSelectMachine }: FactoryMapProps) {
  const getFillColor = (status: string) => {
    switch (status) {
      case 'critical': return '#ef4444'; // red-500
      case 'warning': return '#f59e0b'; // amber-500
      case 'ok': return '#10b981'; // emerald-500
      default: return '#334155'; // slate-700
    }
  };

  const nodes = [
    { id: 'CNC-01', cx: 100, cy: 100 },
    { id: 'CNC-02', cx: 100, cy: 300 },
    { id: 'STAMP-01', cx: 300, cy: 200 },
    { id: 'WELD-01', cx: 500, cy: 200 },
  ];

  return (
    <div className="bg-surface rounded-lg p-4 border border-slate-700 shadow-xl overflow-hidden relative h-full">
      <h2 className="text-xl font-bold mb-4 text-slate-100">Factory Floor</h2>
      <svg width="100%" height="100%" viewBox="0 0 600 400" className="drop-shadow-md">
        {/* Edges */}
        <line x1="100" y1="100" x2="300" y2="200" stroke="#475569" strokeWidth="4" />
        <line x1="100" y1="300" x2="300" y2="200" stroke="#475569" strokeWidth="4" />
        <line x1="300" y1="200" x2="500" y2="200" stroke="#475569" strokeWidth="4" />
        
        {/* Nodes */}
        {nodes.map(node => {
          const machine = machines[node.id];
          const isSelected = selectedMachineId === node.id;
          return (
            <g 
              key={node.id} 
              transform={`translate(${node.cx}, ${node.cy})`}
              onClick={() => onSelectMachine(node.id)}
              className="cursor-pointer transition-transform hover:scale-110 duration-200"
            >
              <circle 
                r={isSelected ? 45 : 40} 
                fill={getFillColor(machine?.status || 'ok')} 
                stroke={isSelected ? '#38bdf8' : '#1e293b'} 
                strokeWidth={isSelected ? 4 : 2}
                className={clsx(
                  machine?.status === 'critical' && "animate-pulse-fast"
                )}
              />
              <text 
                textAnchor="middle" 
                y="5" 
                fill="#ffffff" 
                className="text-xs font-semibold pointer-events-none"
              >
                {node.id}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  );
}
