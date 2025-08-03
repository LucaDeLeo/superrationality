import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { renderHook, act } from '@testing-library/react'
import { ChartTooltip, useChartTooltip } from '../ChartTooltip'

describe('ChartTooltip', () => {
  it('renders null when no data is provided', () => {
    const { container } = render(<ChartTooltip data={null} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders tooltip content when data is provided', () => {
    const tooltipData = {
      x: 100,
      y: 200,
      content: <div>Tooltip Content</div>,
    }
    
    render(<ChartTooltip data={tooltipData} />)
    expect(screen.getByText('Tooltip Content')).toBeInTheDocument()
  })

  it('positions tooltip at specified coordinates', () => {
    const tooltipData = {
      x: 150,
      y: 250,
      content: 'Test Tooltip',
    }
    
    render(<ChartTooltip data={tooltipData} />)
    const tooltip = screen.getByText('Test Tooltip').parentElement
    
    expect(tooltip).toHaveStyle({
      left: '150px',
      top: '250px',
    })
  })

  it('applies theme styles to tooltip', () => {
    const tooltipData = {
      x: 0,
      y: 0,
      content: 'Styled Tooltip',
    }
    
    render(<ChartTooltip data={tooltipData} />)
    const tooltip = screen.getByText('Styled Tooltip').parentElement
    
    expect(tooltip).toHaveClass('fixed z-50 pointer-events-none')
    expect(tooltip).toHaveStyle({
      backgroundColor: '#1f2937',
      color: '#ffffff',
    })
  })
})

describe('useChartTooltip', () => {
  it('initializes with null tooltip data', () => {
    const { result } = renderHook(() => useChartTooltip())
    
    const { container } = render(result.current.tooltip)
    expect(container.firstChild).toBeNull()
  })

  it('shows tooltip when showTooltip is called', () => {
    const { result } = renderHook(() => useChartTooltip())
    
    act(() => {
      result.current.showTooltip(100, 200, 'Test Content')
    })
    
    const { container } = render(result.current.tooltip)
    expect(screen.getByText('Test Content')).toBeInTheDocument()
  })

  it('hides tooltip when hideTooltip is called', () => {
    const { result } = renderHook(() => useChartTooltip())
    
    act(() => {
      result.current.showTooltip(100, 200, 'Test Content')
    })
    
    act(() => {
      result.current.hideTooltip()
    })
    
    const { container } = render(result.current.tooltip)
    expect(container.firstChild).toBeNull()
  })
})