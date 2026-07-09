import React from 'react';
import clsx from 'clsx';
import { Alert } from '../hooks/useAnomalyFeed';
import { AlertCircle, AlertTriangle, Info } from 'lucide-react';

interface AlertFeedProps {
  alerts: Alert[];
}

export function AlertFeed({ alerts }: AlertFeedProps) {
  return (
    <div className="bg-surface rounded-lg p-4 border border-slate-700 shadow-xl flex flex-col h-full max-h-[400px]">
      <h2 className="text-xl font-bold mb-4 text-slate-100 flex items-center gap-2">
        <AlertCircle size={20} className="text-accent" />
        Live Alerts
      </h2>
      
      <div className="flex-1 overflow-y-auto space-y-3 pr-2 custom-scrollbar">
        {alerts.length === 0 ? (
          <div className="text-slate-400 text-sm text-center py-4">No recent alerts.</div>
        ) : (
          alerts.map(alert => (
            <div 
              key={alert.id}
              className={clsx(
                "p-3 rounded border border-l-4 text-sm animate-fade-in",
                alert.severity === 'critical' ? "border-critical bg-critical/10 text-slate-200" :
                alert.severity === 'warning' ? "border-warning bg-warning/10 text-slate-200" :
                "border-primary bg-primary/10 text-slate-200"
              )}
            >
              <div className="flex justify-between items-start mb-1">
                <span className="font-bold flex items-center gap-1">
                  {alert.severity === 'critical' && <AlertCircle size={14} className="text-critical" />}
                  {alert.severity === 'warning' && <AlertTriangle size={14} className="text-warning" />}
                  {alert.severity === 'info' && <Info size={14} className="text-primary" />}
                  {alert.machineId}
                </span>
                <span className="text-xs opacity-75">
                  {new Date(alert.timestamp).toLocaleTimeString()}
                </span>
              </div>
              <p className="opacity-90">{alert.message}</p>
              {alert.severity !== 'info' && (
                <div className="mt-2 text-xs font-mono bg-black/20 p-1 rounded inline-block">
                  Risk Score: {alert.score.toFixed(3)}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
