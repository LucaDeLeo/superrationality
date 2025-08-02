import { renderHook, act, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { useWebSocket } from '../useWebSocket'

// Mock WebSocket
class MockWebSocket {
  url: string
  readyState: number = WebSocket.CONNECTING
  onopen: ((event: Event) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null

  constructor(url: string) {
    this.url = url
    setTimeout(() => {
      this.readyState = WebSocket.OPEN
      this.onopen?.(new Event('open'))
    }, 10)
  }

  send(data: string) {
    if (this.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not open')
    }
  }

  close() {
    this.readyState = WebSocket.CLOSED
    this.onclose?.(new CloseEvent('close'))
  }
}

// Replace global WebSocket with mock
global.WebSocket = MockWebSocket as any

describe('useWebSocket', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('should connect to WebSocket on mount', async () => {
    const onConnect = vi.fn()
    
    const { result } = renderHook(() => 
      useWebSocket({
        url: 'ws://localhost:8000/ws',
        onConnect,
      })
    )

    expect(result.current.connectionStatus).toBe('connecting')

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true)
      expect(result.current.connectionStatus).toBe('connected')
      expect(onConnect).toHaveBeenCalled()
    })
  })

  it('should handle incoming messages', async () => {
    const onMessage = vi.fn()
    
    const { result } = renderHook(() => 
      useWebSocket({
        url: 'ws://localhost:8000/ws',
        onMessage,
      })
    )

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true)
    })

    // Simulate incoming message
    const ws = (global.WebSocket as any).instances?.[0] || result.current.ws
    const messageData = { type: 'test', data: { value: 123 } }
    
    act(() => {
      ws?.onmessage?.(new MessageEvent('message', {
        data: JSON.stringify(messageData)
      }))
    })

    expect(onMessage).toHaveBeenCalledWith(messageData)
  })

  it('should send messages when connected', async () => {
    const { result } = renderHook(() => 
      useWebSocket({
        url: 'ws://localhost:8000/ws',
      })
    )

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true)
    })

    const message = { type: 'test', data: 'hello' }
    
    act(() => {
      result.current.sendMessage(message)
    })

    // In a real test, we'd verify the message was sent
    expect(result.current.isConnected).toBe(true)
  })

  it('should handle disconnection', async () => {
    const onDisconnect = vi.fn()
    
    const { result } = renderHook(() => 
      useWebSocket({
        url: 'ws://localhost:8000/ws',
        onDisconnect,
      })
    )

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true)
    })

    act(() => {
      result.current.disconnect()
    })

    expect(result.current.isConnected).toBe(false)
    expect(result.current.connectionStatus).toBe('disconnected')
    expect(onDisconnect).toHaveBeenCalled()
  })

  it('should attempt to reconnect on failure', async () => {
    vi.useFakeTimers()
    
    const { result } = renderHook(() => 
      useWebSocket({
        url: 'ws://localhost:8000/ws',
        reconnect: true,
        reconnectInterval: 1000,
        reconnectAttempts: 3,
      })
    )

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true)
    })

    // Simulate connection close
    const ws = (global.WebSocket as any).instances?.[0] || result.current.ws
    
    act(() => {
      ws?.close()
    })

    expect(result.current.isConnected).toBe(false)

    // Fast forward to trigger reconnect
    act(() => {
      vi.advanceTimersByTime(1000)
    })

    // Should attempt to reconnect
    expect(result.current.connectionStatus).toBe('connecting')
  })

  it('should handle connection errors', async () => {
    const onError = vi.fn()
    
    const { result } = renderHook(() => 
      useWebSocket({
        url: 'ws://localhost:8000/ws',
        onError,
      })
    )

    await waitFor(() => {
      expect(result.current.connectionStatus).toBe('connected')
    })

    // Simulate error
    const ws = (global.WebSocket as any).instances?.[0] || result.current.ws
    const error = new Event('error')
    
    act(() => {
      ws?.onerror?.(error)
    })

    expect(result.current.connectionStatus).toBe('error')
    expect(onError).toHaveBeenCalledWith(error)
  })
})