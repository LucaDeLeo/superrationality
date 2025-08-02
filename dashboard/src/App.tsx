import React, { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { ThemeProvider } from '@components/theme/ThemeProvider'
import { Layout } from '@components/layout/Layout'
import { Dashboard } from '@pages/Dashboard'
import { NotFound } from '@pages/errors/NotFound'
import { ServerError } from '@pages/errors/ServerError'
import { ErrorBoundary } from '@components/error/ErrorBoundary'
import { WebSocketProvider } from '@/contexts/WebSocketContext'
import { Login } from '@pages/Login'
import { api } from '@services/api'
import './App.css'

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Check if user is already authenticated
    const checkAuth = async () => {
      const token = localStorage.getItem('auth_token')
      if (token) {
        try {
          await api.getCurrentUser()
          setIsAuthenticated(true)
        } catch (error) {
          // Token is invalid, remove it
          localStorage.removeItem('auth_token')
          setIsAuthenticated(false)
        }
      }
      setIsLoading(false)
    }
    checkAuth()
  }, [])

  const handleLogin = () => {
    setIsAuthenticated(true)
  }

  const handleLogout = async () => {
    try {
      await api.logout()
    } catch (error) {
      console.error('Logout error:', error)
    } finally {
      setIsAuthenticated(false)
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  return (
    <ThemeProvider>
      <ErrorBoundary>
        <Router>
          <Routes>
            {/* Public routes */}
            <Route
              path="/login"
              element={
                isAuthenticated ? (
                  <Navigate to="/" replace />
                ) : (
                  <Login onLogin={handleLogin} />
                )
              }
            />

            {/* Protected routes */}
            <Route
              path="/*"
              element={
                isAuthenticated ? (
                  <WebSocketProvider>
                    <Layout onLogout={handleLogout}>
                      <Routes>
                        <Route path="/" element={<Dashboard />} />
                        <Route path="/500" element={<ServerError />} />
                        <Route path="*" element={<NotFound />} />
                      </Routes>
                    </Layout>
                  </WebSocketProvider>
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
          </Routes>
        </Router>
      </ErrorBoundary>
    </ThemeProvider>
  )
}

export default App
