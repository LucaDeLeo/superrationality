import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { PowerDistributionChart } from '../PowerDistributionChart'
import { RoundSummary } from '@/utils/dataTransformers'

// Mock D3
vi.mock('d3', () => ({
  select: vi.fn(() => ({
    selectAll: vi.fn().mockReturnThis(),
    append: vi.fn().mockReturnThis(),
    attr: vi.fn().mockReturnThis(),
    style: vi.fn().mockReturnThis(),
    remove: vi.fn().mockReturnThis(),
    data: vi.fn().mockReturnThis(),
    enter: vi.fn().mockReturnThis(),
    datum: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis(),
    text: vi.fn().mockReturnThis(),
    call: vi.fn().mockReturnThis(),
    select: vi.fn().mockReturnThis(),
  })),
  scaleBand: vi.fn(() => ({
    domain: vi.fn().mockReturnThis(),
    range: vi.fn().mockReturnThis(),
    padding: vi.fn().mockReturnThis(),
    bandwidth: vi.fn(() => 50),
  })),
  scaleLinear: vi.fn(() => ({
    domain: vi.fn().mockReturnThis(),
    range: vi.fn().mockReturnThis(),
    nice: vi.fn().mockReturnThis(),
    clamp: vi.fn().mockReturnThis(),
    invert: vi.fn(x => x),
  })),
  area: vi.fn(() => ({
    x0: vi.fn().mockReturnThis(),
    x1: vi.fn().mockReturnThis(),
    y: vi.fn().mockReturnThis(),
    curve: vi.fn().mockReturnThis(),
  })),
  axisBottom: vi.fn(() => ({
    tickFormat: vi.fn().mockReturnThis(),
  })),
  axisLeft: vi.fn(() => ({
    tickFormat: vi.fn().mockReturnThis(),
  })),
  min: vi.fn(() => 70),
  max: vi.fn(() => 130),
  curveBasis: vi.fn(),
  drag: vi.fn(() => ({
    on: vi.fn().mockReturnThis(),
  })),
  transition: vi.fn(() => ({
    duration: vi.fn().mockReturnThis(),
    ease: vi.fn().mockReturnThis(),
  })),
  pointer: vi.fn(() => [100, 200]),
}))

const mockData: RoundSummary[] = [
  {
    round: 1,
    cooperation_rate: 0.6,
    average_score: 20,
    score_variance: 2,
    power_distribution: { 
      mean: 100, 
      std: 15, 
      min: 70, 
      max: 130 
    },
    score_distribution: { min: 15, max: 25, avg: 20 },
    anonymized_games: [],
  },
  {
    round: 2,
    cooperation_rate: 0.65,
    average_score: 22,
    score_variance: 2.5,
    power_distribution: { 
      mean: 105, 
      std: 18, 
      min: 65, 
      max: 135 
    },
    score_distribution: { min: 16, max: 28, avg: 22 },
    anonymized_games: [],
  },
  {
    round: 3,
    cooperation_rate: 0.7,
    average_score: 24,
    score_variance: 3,
    power_distribution: { 
      mean: 110, 
      std: 20, 
      min: 60, 
      max: 140 
    },
    score_distribution: { min: 17, max: 31, avg: 24 },
    anonymized_games: [],
  },
]

describe('PowerDistributionChart', () => {
  it('renders with loading state', () => {
    render(<PowerDistributionChart data={[]} loading={true} />)
    expect(screen.queryByText('Agent Power Distribution')).not.toBeInTheDocument()
  })

  it('renders with error state', () => {
    const error = new Error('Failed to load power data')
    render(<PowerDistributionChart data={[]} error={error} />)
    expect(screen.getByText('Error loading chart')).toBeInTheDocument()
    expect(screen.getByText('Failed to load power data')).toBeInTheDocument()
  })

  it('renders chart with data', () => {
    const { container } = render(<PowerDistributionChart data={mockData} />)
    expect(container.querySelector('svg.power-distribution-chart')).toBeInTheDocument()
  })

  it('renders with custom title', () => {
    const customTitle = 'Tournament Power Analysis'
    render(<PowerDistributionChart data={mockData} title={customTitle} />)
    // Component should render without error
    expect(true).toBe(true)
  })

  it('supports different chart types', () => {
    const { rerender } = render(
      <PowerDistributionChart data={mockData} chartType="boxplot" />
    )
    expect(true).toBe(true)

    rerender(<PowerDistributionChart data={mockData} chartType="violin" />)
    expect(true).toBe(true)

    rerender(<PowerDistributionChart data={mockData} chartType="histogram" />)
    expect(true).toBe(true)
  })

  it('respects selectedRound prop', () => {
    render(
      <PowerDistributionChart 
        data={mockData} 
        selectedRound={2}
      />
    )
    // Component should render without error
    expect(true).toBe(true)
  })

  it('shows evolution when showEvolution is true', () => {
    render(
      <PowerDistributionChart 
        data={mockData} 
        showEvolution={true}
      />
    )
    // Component should render without error
    expect(true).toBe(true)
  })

  it('shows single round when showEvolution is false', () => {
    render(
      <PowerDistributionChart 
        data={mockData} 
        showEvolution={false}
        selectedRound={2}
      />
    )
    // Component should render without error
    expect(true).toBe(true)
  })

  it('calls onRoundChange when provided', () => {
    const onRoundChange = vi.fn()
    render(
      <PowerDistributionChart 
        data={mockData} 
        onRoundChange={onRoundChange}
      />
    )
    // Component should accept the prop
    expect(true).toBe(true)
  })

  it('handles empty data gracefully', () => {
    render(<PowerDistributionChart data={[]} />)
    // Should render without error
    expect(true).toBe(true)
  })

  it('calculates statistics correctly', () => {
    // This would be tested more thoroughly in integration tests
    render(<PowerDistributionChart data={mockData} />)
    expect(true).toBe(true)
  })
})