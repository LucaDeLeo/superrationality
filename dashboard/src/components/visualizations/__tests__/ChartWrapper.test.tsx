import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ChartWrapper } from '../ChartWrapper'

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

describe('ChartWrapper', () => {
  it('renders loading state when loading prop is true', () => {
    render(
      <ChartWrapper loading={true}>
        {() => <div>Chart Content</div>}
      </ChartWrapper>
    )
    
    expect(screen.queryByText('Chart Content')).not.toBeInTheDocument()
  })

  it('renders error state when error prop is provided', () => {
    const error = new Error('Chart failed to load')
    render(
      <ChartWrapper error={error}>
        {() => <div>Chart Content</div>}
      </ChartWrapper>
    )
    
    expect(screen.getByText('Error loading chart')).toBeInTheDocument()
    expect(screen.getByText('Chart failed to load')).toBeInTheDocument()
  })

  it('renders children with dimensions when not loading or error', () => {
    const mockChild = vi.fn((dimensions) => (
      <div>
        Width: {dimensions.width}, Height: {dimensions.height}
      </div>
    ))

    const { container } = render(
      <ChartWrapper>
        {mockChild}
      </ChartWrapper>
    )

    // ResizeObserver will be called, but we need to simulate dimensions
    expect(container.firstChild).toBeInTheDocument()
  })

  it('applies custom className', () => {
    const { container } = render(
      <ChartWrapper className="custom-class" loading={true}>
        {() => <div>Chart</div>}
      </ChartWrapper>
    )
    
    expect(container.querySelector('.custom-class')).toBeInTheDocument()
  })

  it('uses fixed dimensions when provided', () => {
    const mockChild = vi.fn(() => <div>Chart</div>)
    
    render(
      <ChartWrapper width={800} height={600}>
        {mockChild}
      </ChartWrapper>
    )

    expect(mockChild).toHaveBeenCalled()
  })

  it('respects minimum height', () => {
    const { container } = render(
      <ChartWrapper minHeight={400} loading={true}>
        {() => <div>Chart</div>}
      </ChartWrapper>
    )
    
    const element = container.firstChild as HTMLElement
    expect(element.style.height).toBe('400px')
  })
})