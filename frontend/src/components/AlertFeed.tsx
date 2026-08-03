import React from 'react';
import clsx from 'clsx';
import { Alert } from '../hooks/useAnomalyFeed';
import { AlertCircle, AlertTriangle, Info, MapPin } from 'lucide-react';

interface AlertFeedProps {
  alerts: Alert[];
}

export function AlertFeed({ alerts }: AlertFeedProps) {
  // Group alerts by zone
  const groupedAlerts = alerts.reduce((acc, alert) => {
    const zone = alert.zone || 'General Zone';
    if (!acc[zone]) acc[zone] = [];
    acc[zone].push(alert);
    return acc;
  }, {} as Record<string, Alert[]>);

  return (
    <div className="bg-surface rounded-lg p-4 border border-slate-700 shadow-xl flex flex-col h-full max-h-[400px]">
      <h2 className="text-xl font-bold mb-4 text-slate-100 flex items-center gap-2">
        <AlertCircle size={20} className="text-accent" />
        Live Alerts
      </h2>
      
      <div className="flex-1 overflow-y-auto space-y-4 pr-2 custom-scrollbar">
        {alerts.length === 0 ? (
          <div className="text-slate-400 text-sm text-center py-4">No recent alerts.</div>
        ) : (
          Object.entries(groupedAlerts).map(([zone, zoneAlerts]) => (
            <div key={zone} className="mb-4">
              <h3 className="text-sm font-bold text-slate-300 mb-2 flex items-center gap-1 border-b border-slate-700 pb-1">
                <MapPin size={14} className="text-slate-400" />
                {zone}
              </h3>
              <div className="space-y-3">
                {zoneAlerts.map(alert => (
                  <div 
                    key={alert.id}
                    className={clsx(
                      "p-3 rounded border border-l-4 text-sm animate-fade-in relative",
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
                    <div className="flex justify-between items-center mt-2">
                      {alert.severity !== 'info' && (
                        <div className="text-xs font-mono bg-black/20 p-1 rounded inline-block">
                          Risk Score: {alert.score.toFixed(3)}
                        </div>
                      )}
                      {alert.duplicateCount && alert.duplicateCount > 1 && (
                        <div className="text-xs font-bold text-accent bg-accent/20 px-2 py-0.5 rounded-full border border-accent/30 animate-pulse">
                          Same fault seen {alert.duplicateCount} times
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
