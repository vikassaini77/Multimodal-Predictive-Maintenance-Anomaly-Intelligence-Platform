import { useState, useEffect, useCallback } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';

export type Severity = 'info' | 'warning' | 'critical';

export interface Alert {
  id: string;
  machineId: string;
  timestamp: string;
  score: number;
  message: string;
  severity: Severity;
}

export interface MachineStats {
  id: string;
  name: string;
  currentScore: number;
  history: { time: string; score: number }[];
  status: 'ok' | 'warning' | 'critical';
}

export function useAnomalyFeed(socketUrl: string) {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [machines, setMachines] = useState<Record<string, MachineStats>>({
    'CNC-01': { id: 'CNC-01', name: 'Milling CNC A', currentScore: 0.1, history: [], status: 'ok' },
    'CNC-02': { id: 'CNC-02', name: 'Milling CNC B', currentScore: 0.2, history: [], status: 'ok' },
    'STAMP-01': { id: 'STAMP-01', name: 'Stamping Press', currentScore: 0.15, history: [], status: 'ok' },
    'WELD-01': { id: 'WELD-01', name: 'Robotic Welder', currentScore: 0.05, history: [], status: 'ok' },
  });

  const { lastJsonMessage, readyState } = useWebSocket(socketUrl, {
    shouldReconnect: () => true,
    reconnectAttempts: 10,
    reconnectInterval: 3000,
  });

  useEffect(() => {
    if (lastJsonMessage !== null) {
      const data = lastJsonMessage as any;
      
      // Handle alert event
      if (data.type === 'alert') {
        const newAlert: Alert = {
          id: data.id || Math.random().toString(36).substring(7),
          machineId: data.machineId,
          timestamp: new Date().toISOString(),
          score: data.score,
          message: data.message,
          severity: data.score > 0.8 ? 'critical' : data.score > 0.5 ? 'warning' : 'info'
        };
        setAlerts((prev) => [newAlert, ...prev].slice(0, 50));
      }
      
      // Handle score update
      if (data.type === 'score_update') {
        setMachines((prev) => {
          const machine = prev[data.machineId];
          if (!machine) return prev;
          
          const newHistory = [...machine.history, { time: new Date().toLocaleTimeString(), score: data.score }].slice(-24);
          let status: 'ok' | 'warning' | 'critical' = 'ok';
          if (data.score > 0.8) status = 'critical';
          else if (data.score > 0.5) status = 'warning';
          
          return {
            ...prev,
            [data.machineId]: {
              ...machine,
              currentScore: data.score,
              history: newHistory,
              status
            }
          };
        });
      }
    }
  }, [lastJsonMessage]);

  const connectionStatus = {
    [ReadyState.CONNECTING]: 'Connecting',
    [ReadyState.OPEN]: 'Open',
    [ReadyState.CLOSING]: 'Closing',
    [ReadyState.CLOSED]: 'Closed',
    [ReadyState.UNINSTANTIATED]: 'Uninstantiated',
  }[readyState];

  return { alerts, machines, connectionStatus, readyState };
}
