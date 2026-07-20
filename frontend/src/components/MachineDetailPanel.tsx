import React, { useState, useEffect } from 'react';
import { MachineStats } from '../hooks/useAnomalyFeed';
import { Activity, Camera, Cpu, Loader2, Play } from 'lucide-react';
import clsx from 'clsx';

interface MachineDetailPanelProps {
  machine: MachineStats | null;
}

export function MachineDetailPanel({ machine }: MachineDetailPanelProps) {
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState<any>(null);

  useEffect(() => {
    // Reset prediction state when machine changes
    setIsPredicting(false);
    setPredictionResult(null);
  }, [machine?.id]);

  if (!machine) {
    return (
      <div className="bg-surface rounded-lg p-4 border border-slate-700 shadow-xl h-full min-h-[300px] flex items-center justify-center text-slate-400">
        <div className="text-center">
          <Cpu size={32} className="mx-auto mb-2 opacity-50" />
          <p>No machine selected.</p>
        </div>
      </div>
    );
  }

  const handlePredict = async () => {
    setIsPredicting(true);
    setPredictionResult(null);
    try {
      const payload = {
        machine_id: machine.id,
        timestamp: Date.now() / 1000,
        sensor_data: Array(14).fill(Array(10).fill(0.5)),
        visual_data: Array(3).fill(Array(224).fill(Array(224).fill(0.1))),
        graph: {
            nodes: [{ id: machine.id, type: "machine", features: Array(64).fill(0.0) }],
            edges: []
        }
      };

      const res = await fetch('/api/graph/predict/full', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      const data = await res.json();
      const jobId = data.job_id;

      // Poll for job status
      const interval = setInterval(async () => {
        try {
          const statusRes = await fetch(`/api/graph/jobs/${jobId}`);
          const statusData = await statusRes.json();
          if (statusData.status === 'success') {
            clearInterval(interval);
            setPredictionResult(statusData.result);
            setIsPredicting(false);
          } else if (statusData.status === 'failure' || statusData.status === 'revoked') {
            clearInterval(interval);
            setIsPredicting(false);
            console.error("Job failed:", statusData);
          }
        } catch (err) {
          clearInterval(interval);
          setIsPredicting(false);
          console.error("Polling error:", err);
        }
      }, 1000);

    } catch (err) {
      console.error(err);
      setIsPredicting(false);
    }
  };

  return (
    <div className="bg-surface rounded-lg p-4 border border-slate-700 shadow-xl h-full min-h-[300px] flex flex-col relative overflow-hidden">
      <div className="flex justify-between items-start mb-4 pb-4 border-b border-slate-700">
        <div>
          <h2 className="text-xl font-bold text-slate-100">{machine.name}</h2>
          <p className="text-sm text-slate-400 font-mono">ID: {machine.id}</p>
        </div>
        <div className="text-right">
          <div className="text-2xl font-mono font-bold text-slate-100">
            {predictionResult ? (predictionResult.anomaly_score * 100).toFixed(1) : (machine.currentScore * 100).toFixed(1)}%
          </div>
          <p className="text-xs text-slate-400">{predictionResult ? "GNN Calibrated Risk" : "Current Risk"}</p>
        </div>
      </div>
      
      <div className="mb-4">
        <button 
          onClick={handlePredict}
          disabled={isPredicting}
          className="w-full bg-primary/20 hover:bg-primary/30 text-primary border border-primary/50 font-semibold py-2 px-4 rounded flex items-center justify-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isPredicting ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
          {isPredicting ? "Running Full Diagnostics..." : "Run Full Diagnostics"}
        </button>
      </div>

      <div className="grid grid-cols-2 gap-4 flex-1">
        {/* Sensor Data Mock */}
        <div className="bg-background rounded p-3 border border-slate-700/50 flex flex-col">
          <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1">
            <Activity size={14} /> Telemetry
          </h4>
          <div className="flex-1 flex flex-col justify-center space-y-2 font-mono text-sm">
            <div className="flex justify-between">
              <span className="text-slate-500">Vibration:</span>
              <span className="text-slate-200">2.4 mm/s</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Acoustic:</span>
              <span className="text-slate-200">84 dB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Temp:</span>
              <span className="text-slate-200">68 °C</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Spindle Load:</span>
              <span className="text-slate-200">72 %</span>
            </div>
          </div>
        </div>

        {/* Visual Data Mock */}
        <div className="bg-background rounded p-3 border border-slate-700/50 flex flex-col">
          <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1">
            <Camera size={14} /> Camera Feed
          </h4>
          <div className="flex-1 bg-slate-800 rounded relative overflow-hidden flex items-center justify-center border border-slate-700">
            {/* Mocking a GradCAM overlay using CSS gradients for visual effect */}
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-blue-900/40 via-slate-900/80 to-slate-900"></div>
            
            {machine.status === 'critical' && (
              <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-red-600/50 via-transparent to-transparent opacity-80 animate-pulse"></div>
            )}
            
            {machine.status === 'warning' && (
              <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-amber-500/40 via-transparent to-transparent opacity-80 animate-pulse"></div>
            )}
            
            <span className="relative text-xs text-slate-500 font-mono bg-black/40 px-2 py-1 rounded backdrop-blur-sm border border-slate-600/50">
              Live Feed Active
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
