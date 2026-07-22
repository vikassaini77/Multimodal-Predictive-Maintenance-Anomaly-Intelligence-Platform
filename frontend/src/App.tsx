import React, { useState, useEffect } from 'react';
import { useAnomalyFeed } from './hooks/useAnomalyFeed';
import { FactoryMap } from './components/FactoryMap';
import { AlertFeed } from './components/AlertFeed';
import { RiskScoreChart } from './components/RiskScoreChart';
import { MachineDetailPanel } from './components/MachineDetailPanel';
import { AgentChatPanel } from './components/chat/AgentChatPanel';
import { FailedJobsPanel } from './components/FailedJobsPanel';
import { Login } from './components/Login';
import { Activity, Radio, MessageSquareText, AlertOctagon, X, LogOut } from 'lucide-react';
import clsx from 'clsx';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const { alerts, machines, connectionStatus } = useAnomalyFeed('ws://localhost:8000/ws/edge-feed');
  const [selectedMachineId, setSelectedMachineId] = useState<string | null>(null);
  const [showChat, setShowChat] = useState(false);
  const [toastAlert, setToastAlert] = useState<{ id: string, message: string } | null>(null);

  useEffect(() => {
    // Connect to alerts WebSocket
    const ws = new WebSocket('ws://localhost:8000/graph/ws/alerts');
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'dlq_alert') {
          setToastAlert({
            id: data.task_id,
            message: `Background Job Failed: ${data.error}`
          });
          // Dispatch custom event to tell FailedJobsPanel to refresh
          window.dispatchEvent(new Event('refresh-dlq'));
          
          // Auto-hide toast after 5s
          setTimeout(() => setToastAlert(null), 5000);
        }
      } catch (err) {
        console.error("Failed to parse alert", err);
      }
    };

    return () => {
      ws.close();
    };
  }, [isAuthenticated]);

  const handleLogout = async () => {
    try {
      await fetch('/api/auth/logout', { method: 'POST' });
    } catch (e) {
      console.error(e);
    }
    setIsAuthenticated(false);
  };

  if (!isAuthenticated) {
    return <Login onSuccess={() => setIsAuthenticated(true)} />;
  }

  const selectedMachine = selectedMachineId ? machines[selectedMachineId] : null;

  return (
    <div className="min-h-screen bg-background text-slate-200 p-6 font-sans flex flex-col relative overflow-hidden">
      {/* Toast Notification */}
      {toastAlert && (
        <div className="absolute top-4 right-4 z-50 animate-slide-in-right">
          <div className="bg-red-900 border border-red-500 text-white px-4 py-3 rounded-lg shadow-2xl flex items-start gap-3 max-w-sm">
            <AlertOctagon className="shrink-0 mt-0.5 text-red-300" size={20} />
            <div className="flex-1">
              <h4 className="font-bold text-sm">Task Moved to DLQ</h4>
              <p className="text-xs text-red-200 mt-1 break-words">{toastAlert.message}</p>
            </div>
            <button onClick={() => setToastAlert(null)} className="text-red-300 hover:text-white">
              <X size={16} />
            </button>
          </div>
        </div>
      )}

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
          
          <button 
            onClick={handleLogout}
            className="flex items-center gap-2 px-4 py-2 rounded-full border border-red-900/50 hover:bg-red-900/20 text-red-400 hover:text-red-300 transition-colors"
          >
            <LogOut size={16} />
            <span className="text-sm font-semibold">Logout</span>
          </button>
        </div>
      </header>

      {/* Main Grid Layout */}
      <div className="flex-1 grid grid-cols-12 gap-6 min-h-[600px] h-[calc(100vh-120px)]">
        
        {/* Left Column: Factory Map & Chart */}
        <div className={clsx(
          "flex flex-col gap-6 transition-all duration-300 h-full overflow-y-auto pr-2",
          showChat ? "col-span-12 lg:col-span-5" : "col-span-12 lg:col-span-8"
        )}>
          <div>
            <FactoryMap 
              machines={machines} 
              selectedMachineId={selectedMachineId} 
              onSelectMachine={setSelectedMachineId} 
            />
          </div>
          <div className={clsx(showChat ? "h-[200px]" : "h-[300px]")}>
            <RiskScoreChart machine={selectedMachine} />
          </div>
          
          <FailedJobsPanel />
        </div>

        {/* Middle Column: Alerts & Details (Hidden or shrinked when chat is open) */}
        <div className={clsx(
          "flex flex-col gap-6 transition-all duration-300 h-full overflow-y-auto pr-2",
          showChat ? "col-span-12 lg:col-span-3" : "col-span-12 lg:col-span-4"
        )}>
          <div className="flex-1 min-h-[300px]">
            <AlertFeed alerts={alerts} />
          </div>
          <div className="shrink-0 h-[320px]">
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
