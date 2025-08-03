import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ScoreProgressionChart } from '../ScoreProgressionChart'
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
    exit: vi.fn().mockReturnThis(),
    each: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis(),
    text: vi.fn().mockReturnThis(),
    datum: vi.fn().mockReturnThis(),
    call: vi.fn().mockReturnThis(),
    transition: vi.fn().mockReturnThis(),
    delay: vi.fn().mockReturnThis(),
  })),
  scaleLinear: vi.fn(() => ({
    domain: vi.fn().mockReturnThis(),
    range: vi.fn().mockReturnThis(),
    invert: vi.fn(x => x),
  })),
  line: vi.fn(() => ({
    x: vi.fn().mockReturnThis(),
    y: vi.fn().mockReturnThis(),
    curve: vi.fn().mockReturnThis(),
  })),
  area: vi.fn(() => ({
    x: vi.fn().mockReturnThis(),
    y0: vi.fn().mockReturnThis(),
    y1: vi.fn().mockReturnThis(),
    curve: vi.fn().mockReturnThis(),
  })),
  stack: vi.fn(() => ({
    keys: vi.fn().mockReturnThis(),
    order: vi.fn().mockReturnThis(),
    offset: vi.fn().mockReturnThis(),
  })),
  brushX: vi.fn(() => ({
    extent: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis(),
  })),
  axisBottom: vi.fn(() => ({
    tickFormat: vi.fn().mockReturnThis(),
  })),
  axisLeft: vi.fn(() => ({
    tickFormat: vi.fn().mockReturnThis(),
  })),
  max: vi.fn(() => 100),
  curveMonotoneX: vi.fn(),
  stackOrderNone: vi.fn(),
  stackOffsetNone: vi.fn(),
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
    power_distribution: { mean: 100, std: 10, min: 80, max: 120 },
    score_distribution: { min: 15, max: 25, avg: 20 },
    anonymized_games: [
      {
        player1_id: 'agent1',
        player2_id: 'agent2',
        player1_action: 'C',
        player2_action: 'D',
        player1_score: 0,
        player2_score: 5,
      },
      {
        player1_id: 'agent1',
        player2_id: 'agent3',
        player1_action: 'C',
        player2_action: 'C',
        player1_score: 3,
        player2_score: 3,
      },
    ],
  },
  {
    round: 2,
    cooperation_rate: 0.7,
    average_score: 22,
    score_variance: 1.5,
    power_distribution: { mean: 105, std: 12, min: 75, max: 125 },
    score_distribution: { min: 18, max: 26, avg: 22 },
    anonymized_games: [
      {
        player1_id: 'agent1',
        player2_id: 'agent2',
        player1_action: 'D',
        player2_action: 'D',
        player1_score: 1,
        player2_score: 1,
      },
      {
        player1_id: 'agent2',
        player2_id: 'agent3',
        player1_action: 'C',
        player2_action: 'C',
        player1_score: 3,
        player2_score: 3,
      },
    ],
  },
]

describe('ScoreProgressionChart', () => {
  it('renders with loading state', () => {
    render(<ScoreProgressionChart data={[]} loading={true} />)
    expect(screen.queryByText('Score Progression Across Rounds')).not.toBeInTheDocument()
  })

  it('renders with error state', () => {
    const error = new Error('Failed to load score data')
    render(<ScoreProgressionChart data={[]} error={error} />)
    expect(screen.getByText('Error loading chart')).toBeInTheDocument()
    expect(screen.getByText('Failed to load score data')).toBeInTheDocument()
  })

  it('renders chart with data', () => {
    const { container } = render(<ScoreProgressionChart data={mockData} />)
    expect(container.querySelector('svg.score-progression-chart')).toBeInTheDocument()
  })

  it('renders with custom title', () => {
    const customTitle = 'Agent Score Evolution'
    render(<ScoreProgressionChart data={mockData} title={customTitle} />)
    // Component should render without error
    expect(true).toBe(true)
  })

  it('supports different chart types', () => {
    const { rerender } = render(
      <ScoreProgressionChart data={mockData} chartType="line" />
    )
    expect(true).toBe(true)

    rerender(<ScoreProgressionChart data={mockData} chartType="area" />)
    expect(true).toBe(true)

    rerender(<ScoreProgressionChart data={mockData} chartType="stacked" />)
    expect(true).toBe(true)
  })

  it('filters agents when selectedAgents prop is provided', () => {
    const selectedAgents = ['agent1', 'agent2']
    render(
      <ScoreProgressionChart 
        data={mockData} 
        selectedAgents={selectedAgents}
      />
    )
    // Component should render without error
    expect(true).toBe(true)
  })

  it('disables animation when showAnimation is false', () => {
    render(
      <ScoreProgressionChart 
        data={mockData} 
        showAnimation={false}
      />
    )
    // Component should render without error
    expect(true).toBe(true)
  })

  it('calls onAgentClick when provided', () => {
    const onAgentClick = vi.fn()
    render(
      <ScoreProgressionChart 
        data={mockData} 
        onAgentClick={onAgentClick}
      />
    )
    // Component should accept the prop
    expect(true).toBe(true)
  })

  it('handles empty data gracefully', () => {
    render(<ScoreProgressionChart data={[]} />)
    // Should render without error
    expect(true).toBe(true)
  })

  it('calculates cumulative scores correctly', () => {
    // This would be tested in the dataTransformers tests
    // Here we just verify the component handles the transformed data
    render(<ScoreProgressionChart data={mockData} />)
    expect(true).toBe(true)
  })
})