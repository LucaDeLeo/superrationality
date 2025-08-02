import React, { createContext, useContext, useEffect, useState } from 'react'
import { useWebSocket, WebSocketMessage } from '@hooks/useWebSocket'

interface WebSocketContextType {
  isConnected: boolean
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error'
  sendMessage: (message: any) => void
  lastMessage: WebSocketMessage | null
  experimentUpdates: Map<string, any>
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined)

export const useWebSocketContext = () => {
  const context = useContext(WebSocketContext)
  if (!context) {
    throw new Error('useWebSocketContext must be used within a WebSocketProvider')
  }
  return context
}

interface WebSocketProviderProps {
  children: React.ReactNode
  wsUrl?: string
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ 
  children, 
  wsUrl = 'ws://localhost:8000/ws' 
}) => {
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [experimentUpdates, setExperimentUpdates] = useState<Map<string, any>>(new Map())

  const handleMessage = (message: WebSocketMessage) => {
    setLastMessage(message)

    // Handle specific message types
    switch (message.type) {
      case 'experiment_update':
        setExperimentUpdates(prev => {
          const newMap = new Map(prev)
          newMap.set(message.data.experiment_id, message.data)
          return newMap
        })
        break
      
      case 'new_round':
        // Handle new round notifications
        console.log('New round completed:', message.data)
        break
      
      default:
        console.log('Received message:', message)
    }
  }

  const { isConnected, connectionStatus, sendMessage } = useWebSocket({
    url: wsUrl,
    onMessage: handleMessage,
    onConnect: () => console.log('WebSocket connected in context'),
    onDisconnect: () => console.log('WebSocket disconnected in context'),
    onError: (error) => console.error('WebSocket error in context:', error),
  })

  const value: WebSocketContextType = {
    isConnected,
    connectionStatus,
    sendMessage,
    lastMessage,
    experimentUpdates,
  }

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  )
}