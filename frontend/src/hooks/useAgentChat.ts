import { useState, useCallback, useEffect } from 'react';
import useWebSocket from 'react-use-websocket';

export type ChatRole = 'user' | 'agent';

export interface TraceStep {
  type: 'Thought' | 'Action' | 'Observation';
  content: string;
}

export interface Citation {
  chunk_id: string;
  document: string;
  text: string;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  streaming?: boolean;
  traces?: TraceStep[];
  citations?: Citation[];
  imageUrl?: string; // e.g., GradCAM snapshot
}

export function useAgentChat(socketUrl: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isAgentTyping, setIsAgentTyping] = useState(false);

  const { sendMessage, lastJsonMessage } = useWebSocket(socketUrl, {
    shouldReconnect: () => true,
    reconnectAttempts: 10,
    reconnectInterval: 3000,
  });

  const sendQuery = useCallback((query: string) => {
    const userMsg: ChatMessage = {
      id: Math.random().toString(36).substring(7),
      role: 'user',
      content: query
    };
    setMessages(prev => [...prev, userMsg]);
    setIsAgentTyping(true);
    sendMessage(JSON.stringify({ query }));
  }, [sendMessage]);

  useEffect(() => {
    if (lastJsonMessage !== null) {
      const data = lastJsonMessage as any;
      
      setMessages(prev => {
        const lastMsg = prev[prev.length - 1];
        
        // If it's a new agent response or the last message was user's
        if (!lastMsg || lastMsg.role === 'user' || data.type === 'start') {
          setIsAgentTyping(false);
          return [...prev, {
            id: data.id || Math.random().toString(36).substring(7),
            role: 'agent',
            content: data.token || '',
            streaming: data.type === 'stream',
            traces: data.trace ? [data.trace] : [],
            citations: data.citations || [],
            imageUrl: data.imageUrl
          }];
        }
        
        // Append to existing streaming message
        const updatedMsg = { ...lastMsg };
        if (data.type === 'stream' && data.token) {
          updatedMsg.content += data.token;
          updatedMsg.streaming = true;
        }
        
        if (data.trace) {
          updatedMsg.traces = [...(updatedMsg.traces || []), data.trace];
        }
        
        if (data.citations) {
          updatedMsg.citations = data.citations;
        }
        
        if (data.imageUrl) {
          updatedMsg.imageUrl = data.imageUrl;
        }

        if (data.type === 'end') {
          updatedMsg.streaming = false;
        }

        return [...prev.slice(0, -1), updatedMsg];
      });
    }
  }, [lastJsonMessage]);

  return { messages, sendQuery, isAgentTyping };
}
