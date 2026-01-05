import { useEffect, useRef, useState, useCallback } from 'react';

export interface WebSocketMessage {
  event_id: string;
  timestamp: string;
  source: string;
  type: string;
  payload: Record<string, any>;
  run_id?: string;
  correlation_id?: string;
}

interface UseWebSocketOptions {
  /** Whether to automatically connect on mount */
  autoConnect?: boolean;
  /** Reconnect delay in milliseconds */
  reconnectDelay?: number;
  /** Maximum reconnection attempts (0 = infinite) */
  maxReconnectAttempts?: number;
  /** Heartbeat interval in milliseconds */
  heartbeatInterval?: number;
}

interface UseWebSocketReturn {
  /** Last received message */
  lastMessage: WebSocketMessage | null;
  /** Connection state */
  isConnected: boolean;
  /** Error state */
  error: Error | null;
  /** Manually connect */
  connect: () => void;
  /** Manually disconnect */
  disconnect: () => void;
  /** Send a message (currently only used for ping) */
  send: (message: string) => void;
}

/**
 * React hook for WebSocket connections with automatic reconnection.
 *
 * @param url - WebSocket URL (e.g., 'ws://localhost:8081/ws/live')
 * @param options - Configuration options
 * @param onMessage - Callback when a message is received
 */
export function useWebSocket(
  url: string,
  options: UseWebSocketOptions = {},
  onMessage?: (message: WebSocketMessage) => void
): UseWebSocketReturn {
  const {
    autoConnect = true,
    reconnectDelay = 3000,
    maxReconnectAttempts = 0, // infinite by default
    heartbeatInterval = 30000, // 30 seconds
  } = options;

  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const heartbeatIntervalRef = useRef<NodeJS.Timeout>();
  const shouldConnectRef = useRef(autoConnect);

  const cleanup = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const startHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
    }

    heartbeatIntervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send('ping');
      }
    }, heartbeatInterval);
  }, [heartbeatInterval]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    cleanup();
    shouldConnectRef.current = true;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log(`WebSocket connected: ${url}`);
        setIsConnected(true);
        setError(null);
        reconnectAttemptsRef.current = 0;
        startHeartbeat();
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage;

          // Ignore pong messages
          if (message.type === 'pong') {
            return;
          }

          setLastMessage(message);
          onMessage?.(message);
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onerror = (event) => {
        console.error(`WebSocket error: ${url}`, event);
        const err = new Error('WebSocket connection error');
        setError(err);
      };

      ws.onclose = () => {
        console.log(`WebSocket disconnected: ${url}`);
        setIsConnected(false);

        if (heartbeatIntervalRef.current) {
          clearInterval(heartbeatIntervalRef.current);
        }

        // Attempt reconnection if enabled and not manually disconnected
        if (shouldConnectRef.current) {
          const shouldReconnect =
            maxReconnectAttempts === 0 ||
            reconnectAttemptsRef.current < maxReconnectAttempts;

          if (shouldReconnect) {
            reconnectAttemptsRef.current++;
            console.log(
              `Reconnecting in ${reconnectDelay}ms... (attempt ${reconnectAttemptsRef.current})`
            );

            reconnectTimeoutRef.current = setTimeout(() => {
              connect();
            }, reconnectDelay);
          } else {
            setError(new Error('Max reconnection attempts reached'));
          }
        }
      };
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setError(err as Error);
    }
  }, [url, cleanup, reconnectDelay, maxReconnectAttempts, startHeartbeat, onMessage]);

  const disconnect = useCallback(() => {
    shouldConnectRef.current = false;
    cleanup();
    setIsConnected(false);
  }, [cleanup]);

  const send = useCallback((message: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(message);
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }, []);

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      cleanup();
    };
  }, [autoConnect, connect, cleanup]);

  return {
    lastMessage,
    isConnected,
    error,
    connect,
    disconnect,
    send,
  };
}
