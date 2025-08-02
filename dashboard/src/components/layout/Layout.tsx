import React, { useState } from 'react'
import { Header } from './Header'
import { Sidebar } from './Sidebar'
import { MainContent } from './MainContent'

interface LayoutProps {
  children: React.ReactNode
  onLogout?: () => void | Promise<void>
}

export const Layout: React.FC<LayoutProps> = ({ children, onLogout }) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)

  const handleMenuToggle = () => {
    setIsSidebarOpen(!isSidebarOpen)
  }

  const handleSidebarClose = () => {
    setIsSidebarOpen(false)
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      <Header onMenuToggle={handleMenuToggle} isMenuOpen={isSidebarOpen} onLogout={onLogout} />
      <div className="flex">
        <Sidebar isOpen={isSidebarOpen} onClose={handleSidebarClose} />
        <MainContent>{children}</MainContent>
      </div>
    </div>
  )
}