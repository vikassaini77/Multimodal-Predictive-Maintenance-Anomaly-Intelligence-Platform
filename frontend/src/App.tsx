import React, { useState } from 'react';
import { useAnomalyFeed } from './hooks/useAnomalyFeed';
import { FactoryMap } from './components/FactoryMap';
import { AlertFeed } from './components/AlertFeed';
import { RiskScoreChart } from './components/RiskScoreChart';
import { MachineDetailPanel } from './components/MachineDetailPanel';
import { Activity, Radio } from 'lucide-react';
import clsx from 'clsx';

function App() {
  const { alerts, machines, connectionStatus } = useAnomalyFeed('ws://localhost:8000/ws/edge-feed');
  const [selectedMachineId, setSelectedMachineId] = useState<string | null>(null);

  const selectedMachine = selectedMachineId ? machines[selectedMachineId] : null;

  return (
    <div className="min-h-screen bg-background text-slate-200 p-6 font-sans">
      {/* Header */}
      <header className="flex justify-between items-center mb-6 pb-4 border-b border-slate-700">
        <div className="flex items-center gap-3">
          <div className="bg-primary/20 p-2 rounded-lg border border-primary/30">
            <Activity className="text-primary" size={24} />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-100 tracking-tight">Multimodal AI Platform</h1>
            <p className="text-sm text-slate-400">Edge Monitoring Dashboard</p>
          </div>
        </div>
        
        <div className="flex items-center gap-2 bg-surface px-4 py-2 rounded-full border border-slate-700 shadow-inner">
          <span className="text-sm font-semibold text-slate-300">Edge Link:</span>
          <div className="flex items-center gap-1.5">
            <Radio 
              size={14} 
              className={clsx(
                connectionStatus === 'Open' ? "text-ok animate-pulse" : "text-critical"
              )} 
            />
            <span className={clsx(
              "text-xs font-mono font-bold uppercase",
              connectionStatus === 'Open' ? "text-ok" : "text-critical"
            )}>
              {connectionStatus}
            </span>
          </div>
        </div>
      </header>

      {/* Main Grid Layout */}
      <div className="grid grid-cols-12 gap-6 h-[calc(100vh-120px)] min-h-[600px]">
        
        {/* Left Column: Factory Map & Chart */}
        <div className="col-span-12 lg:col-span-8 flex flex-col gap-6">
          <div className="flex-1">
            <FactoryMap 
              machines={machines} 
              selectedMachineId={selectedMachineId} 
              onSelectMachine={setSelectedMachineId} 
            />
          </div>
          <div>
            <RiskScoreChart machine={selectedMachine} />
          </div>
        </div>

        {/* Right Column: Alerts & Details */}
        <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
          <div className="flex-1">
            <AlertFeed alerts={alerts} />
          </div>
          <div className="h-[300px]">
            <MachineDetailPanel machine={selectedMachine} />
          </div>
        </div>
        
      </div>
    </div>
  );
}

export default App;
