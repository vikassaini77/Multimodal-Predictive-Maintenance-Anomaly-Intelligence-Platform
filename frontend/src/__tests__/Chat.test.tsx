import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { AgentChatPanel } from '../components/chat/AgentChatPanel';
import { renderHook } from '@testing-library/react';
import { useAgentChat } from '../hooks/useAgentChat';
import React from 'react';

vi.mock('lucide-react', () => ({
  Bot: () => <div data-testid="icon-bot" />,
  User: () => <div data-testid="icon-user" />,
  Send: () => <div data-testid="icon-send" />,
  Loader2: () => <div data-testid="icon-loader" />,
  ChevronDown: () => <div data-testid="icon-chevron-down" />,
  ChevronRight: () => <div data-testid="icon-chevron-right" />,
  Terminal: () => <div data-testid="icon-terminal" />,
  Brain: () => <div data-testid="icon-brain" />,
  Eye: () => <div data-testid="icon-eye" />,
  BookOpen: () => <div data-testid="icon-book" />,
  Camera: () => <div data-testid="icon-camera" />,
  Sparkles: () => <div data-testid="icon-sparkles" />,
}));

const mockSendMessage = vi.fn();
vi.mock('react-use-websocket', () => ({
  default: (...args: any[]) => ({
    sendMessage: mockSendMessage,
    lastJsonMessage: null,
  })
}));

// Mock scrollIntoView
window.HTMLElement.prototype.scrollIntoView = function() {};

describe('AgentChatPanel UI Tests', () => {
  it('renders initial empty state and prompts', () => {
    render(<AgentChatPanel />);
    expect(screen.getByText('Diagnostic Agent')).toBeDefined();
    expect(screen.getByText('How can I help diagnose?')).toBeDefined();
    expect(screen.getByText('Why is Machine 4 flagged?')).toBeDefined(); // Suggested prompt
  });

  it('allows sending a query', () => {
    render(<AgentChatPanel />);
    const input = screen.getByPlaceholderText('Ask a diagnostic question...');
    fireEvent.change(input, { target: { value: 'Test query' } });
    
    // Using closest form submit since we mocked icons
    const form = input.closest('form');
    fireEvent.submit(form!);
    
    expect(mockSendMessage).toHaveBeenCalledWith(JSON.stringify({ query: 'Test query' }));
  });
});

describe('useAgentChat Hook Logic', () => {
  it('initializes with empty messages', () => {
    const { result } = renderHook(() => useAgentChat('ws://test'));
    expect(result.current.messages.length).toBe(0);
    expect(result.current.isAgentTyping).toBe(false);
  });
});
