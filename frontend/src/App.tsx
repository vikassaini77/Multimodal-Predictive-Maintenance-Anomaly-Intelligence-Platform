import React, { useState } from 'react';
import { useAnomalyFeed } from './hooks/useAnomalyFeed';
import { FactoryMap } from './components/FactoryMap';
import { AlertFeed } from './components/AlertFeed';
import { RiskScoreChart } from './components/RiskScoreChart';
import { MachineDetailPanel } from './components/MachineDetailPanel';
import { AgentChatPanel } from './components/chat/AgentChatPanel';
import { Activity, Radio, MessageSquareText } from 'lucide-react';
import clsx from 'clsx';

function App() {
  const { alerts, machines, connectionStatus } = useAnomalyFeed('ws://localhost:8000/ws/edge-feed');
  const [selectedMachineId, setSelectedMachineId] = useState<string | null>(null);
  const [showChat, setShowChat] = useState(false);

  const selectedMachine = selectedMachineId ? machines[selectedMachineId] : null;

  return (
    <div className="min-h-screen bg-background text-slate-200 p-6 font-sans flex flex-col">
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
        
        <div className="flex items-center gap-4">
          <button 
            onClick={() => setShowChat(!showChat)}
            className={clsx(
              "flex items-center gap-2 px-4 py-2 rounded-full border shadow-inner transition-colors font-semibold text-sm",
              showChat 
                ? "bg-primary text-slate-900 border-primary/50" 
                : "bg-surface text-slate-300 border-slate-700 hover:border-primary/50"
            )}
          >
            <MessageSquareText size={16} />
            {showChat ? "Hide Agent" : "Ask Agent"}
          </button>
          
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
        </div>
      </header>

      {/* Main Grid Layout */}
      <div className="flex-1 grid grid-cols-12 gap-6 min-h-[600px] h-[calc(100vh-120px)]">
        
        {/* Left Column: Factory Map & Chart */}
        <div className={clsx(
          "flex flex-col gap-6 transition-all duration-300",
          showChat ? "col-span-12 lg:col-span-5" : "col-span-12 lg:col-span-8"
        )}>
          <div className="flex-1">
            <FactoryMap 
              machines={machines} 
              selectedMachineId={selectedMachineId} 
              onSelectMachine={setSelectedMachineId} 
            />
          </div>
          <div className={clsx(showChat ? "h-[200px]" : "h-auto")}>
            <RiskScoreChart machine={selectedMachine} />
          </div>
        </div>

        {/* Middle Column: Alerts & Details (Hidden or shrinked when chat is open) */}
        <div className={clsx(
          "flex flex-col gap-6 transition-all duration-300",
          showChat ? "col-span-12 lg:col-span-3" : "col-span-12 lg:col-span-4"
        )}>
          <div className="flex-1">
            <AlertFeed alerts={alerts} />
          </div>
          <div className="h-[300px]">
            <MachineDetailPanel machine={selectedMachine} />
          </div>
        </div>

        {/* Right Column: Chat Panel */}
        {showChat && (
          <div className="col-span-12 lg:col-span-4 h-full animate-fade-in">
            <AgentChatPanel />
          </div>
        )}
        
      </div>
    </div>
  );
}

export default App;
