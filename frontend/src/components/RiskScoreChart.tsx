import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { MachineStats } from '../hooks/useAnomalyFeed';

interface RiskScoreChartProps {
  machine: MachineStats | null;
}

export function RiskScoreChart({ machine }: RiskScoreChartProps) {
  if (!machine) {
    return (
      <div className="bg-surface rounded-lg p-4 border border-slate-700 shadow-xl h-64 flex items-center justify-center text-slate-400">
        Select a machine on the map to view its anomaly history.
      </div>
    );
  }

  return (
    <div className="bg-surface rounded-lg p-4 border border-slate-700 shadow-xl h-64">
      <h3 className="text-lg font-bold mb-4 text-slate-100 flex items-center gap-2">
        <span>{machine.name}</span>
        <span className="text-sm font-normal text-slate-400 font-mono">Risk Score Trend</span>
      </h3>
      
      <div className="h-44 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={machine.history} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis 
              dataKey="time" 
              stroke="#64748b" 
              fontSize={10} 
              tickMargin={10}
              minTickGap={30}
            />
            <YAxis 
              stroke="#64748b" 
              fontSize={10} 
              domain={[0, 1.0]} 
              ticks={[0, 0.5, 0.8, 1.0]} 
            />
            <Tooltip 
              contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }}
              itemStyle={{ color: '#38bdf8' }}
              labelStyle={{ color: '#94a3b8', marginBottom: '4px' }}
            />
            
            <ReferenceLine y={0.5} stroke="#f59e0b" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: 'Warning (0.5)', fill: '#f59e0b', fontSize: 10 }} />
            <ReferenceLine y={0.8} stroke="#ef4444" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: 'Critical (0.8)', fill: '#ef4444', fontSize: 10 }} />
            
            <Line 
              type="monotone" 
              dataKey="score" 
              stroke="#38bdf8" 
              strokeWidth={2} 
              dot={{ r: 3, fill: '#0f172a', strokeWidth: 2 }} 
              activeDot={{ r: 6, fill: '#38bdf8' }}
              animationDuration={500}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
