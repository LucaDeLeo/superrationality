import React from 'react'

interface MainContentProps {
  children: React.ReactNode
}

export const MainContent: React.FC<MainContentProps> = ({ children }) => {
  return (
    <main className="flex-1 lg:ml-64">
      <div className="pt-16">
        <div className="p-4 md:p-6 lg:p-8">
          {children}
        </div>
      </div>
    </main>
  )
}