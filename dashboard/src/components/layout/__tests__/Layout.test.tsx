import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { Layout } from '../Layout'

// Mock react-router-dom NavLink
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    NavLink: ({ children, to, className }: any) => (
      <a href={to} className={typeof className === 'function' ? className({ isActive: false }) : className}>
        {children}
      </a>
    ),
  }
})

const renderLayout = (children: React.ReactNode = <div>Test Content</div>) => {
  return render(
    <BrowserRouter>
      <Layout>{children}</Layout>
    </BrowserRouter>
  )
}

describe('Layout Component', () => {

  it('should render header, sidebar, and content', () => {
    renderLayout()
    
    expect(screen.getByText('Acausal Dashboard')).toBeInTheDocument()
    expect(screen.getByText('Test Content')).toBeInTheDocument()
    expect(screen.getByText('Dashboard')).toBeInTheDocument()
    expect(screen.getByText('Experiments')).toBeInTheDocument()
  })

  it('should toggle sidebar on mobile', () => {
    renderLayout()
    
    const menuButton = screen.getByLabelText('Toggle menu')
    
    // Sidebar should be closed initially on mobile
    const sidebar = screen.getByRole('complementary')
    expect(sidebar).toHaveClass('-translate-x-full')
    
    // Click to open
    fireEvent.click(menuButton)
    expect(sidebar).toHaveClass('translate-x-0')
    
    // Click to close
    fireEvent.click(menuButton)
    expect(sidebar).toHaveClass('-translate-x-full')
  })

  it('should close sidebar when overlay is clicked', () => {
    renderLayout()
    
    const menuButton = screen.getByLabelText('Toggle menu')
    fireEvent.click(menuButton)
    
    // Find and click the overlay
    const overlay = document.querySelector('.bg-black.bg-opacity-50')
    expect(overlay).toBeInTheDocument()
    fireEvent.click(overlay!)
    
    // Sidebar should be closed
    const sidebar = screen.getByRole('complementary')
    expect(sidebar).toHaveClass('-translate-x-full')
  })

  it('should render all navigation items', () => {
    renderLayout()
    
    const navItems = ['Dashboard', 'Experiments', 'Tournaments', 'Analytics', 'Settings']
    navItems.forEach(item => {
      expect(screen.getByText(item)).toBeInTheDocument()
    })
  })

  it('should display user information in sidebar', () => {
    renderLayout()
    
    expect(screen.getByText('Admin User')).toBeInTheDocument()
    expect(screen.getByText('admin@acausal.ai')).toBeInTheDocument()
  })
})

describe('Responsive Layout', () => {
  it('should have proper responsive classes', () => {
    renderLayout()
    
    // Check header is fixed
    const header = screen.getByRole('banner')
    expect(header).toHaveClass('fixed', 'top-0', 'w-full', 'z-50')
    
    // Check main content has proper padding
    const main = screen.getByRole('main')
    expect(main).toHaveClass('flex-1', 'lg:ml-64')
    expect(main.firstChild).toHaveClass('pt-16')
  })

  it('should have mobile-first responsive padding', () => {
    renderLayout()
    
    const contentWrapper = screen.getByRole('main').querySelector('.p-4')
    expect(contentWrapper).toHaveClass('p-4', 'md:p-6', 'lg:p-8')
  })
})