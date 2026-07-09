import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AlertFeed } from '../components/AlertFeed';
import { MachineDetailPanel } from '../components/MachineDetailPanel';
import { renderHook } from '@testing-library/react';
import { useAnomalyFeed } from '../hooks/useAnomalyFeed';
import React from 'react';

// Mock the lucide-react icons to avoid SVG rendering issues in JSDOM
vi.mock('lucide-react', () => ({
  AlertCircle: () => <div data-testid="icon-alert-circle" />,
  AlertTriangle: () => <div data-testid="icon-alert-triangle" />,
  Info: () => <div data-testid="icon-info" />,
  Activity: () => <div data-testid="icon-activity" />,
  Camera: () => <div data-testid="icon-camera" />,
  Cpu: () => <div data-testid="icon-cpu" />,
  Radio: () => <div data-testid="icon-radio" />,
}));

// Mock react-use-websocket
const mockUseWebSocket = vi.fn();
vi.mock('react-use-websocket', () => ({
  default: (...args: any[]) => mockUseWebSocket(...args),
  ReadyState: {
    CONNECTING: 0,
    OPEN: 1,
    CLOSING: 2,
    CLOSED: 3,
    UNINSTANTIATED: 4,
  }
}));

describe('Dashboard Component Smoke Tests', () => {
  it('AlertFeed renders empty state correctly', () => {
    render(<AlertFeed alerts={[]} />);
    expect(screen.getByText('No recent alerts.')).toBeDefined();
  });

  it('AlertFeed renders alerts correctly', () => {
    const mockAlerts = [
      { id: '1', machineId: 'CNC-01', timestamp: new Date().toISOString(), score: 0.9, message: 'High vibration', severity: 'critical' as const }
    ];
    render(<AlertFeed alerts={mockAlerts} />);
    expect(screen.getByText('CNC-01')).toBeDefined();
    expect(screen.getByText('High vibration')).toBeDefined();
    expect(screen.getByText('Risk Score: 0.900')).toBeDefined();
  });

  it('MachineDetailPanel renders empty state correctly', () => {
    render(<MachineDetailPanel machine={null} />);
    expect(screen.getByText('No machine selected.')).toBeDefined();
  });
  
  it('MachineDetailPanel renders machine stats correctly', () => {
    const mockMachine = {
      id: 'STAMP-01',
      name: 'Stamping Press',
      currentScore: 0.45,
      history: [],
      status: 'ok' as const
    };
    render(<MachineDetailPanel machine={mockMachine} />);
    expect(screen.getByText('Stamping Press')).toBeDefined();
    expect(screen.getByText('ID: STAMP-01')).toBeDefined();
    expect(screen.getByText('45.0%')).toBeDefined(); // Current risk formatted
  });
});

describe('useAnomalyFeed Hook Logic', () => {
  it('should initialize with default machines and open status', () => {
    mockUseWebSocket.mockReturnValue({
      lastJsonMessage: null,
      readyState: 1 // OPEN
    });
    
    const { result } = renderHook(() => useAnomalyFeed('ws://test'));
    
    expect(result.current.connectionStatus).toBe('Open');
    expect(result.current.machines['CNC-01']).toBeDefined();
    expect(result.current.alerts.length).toBe(0);
  });
});
