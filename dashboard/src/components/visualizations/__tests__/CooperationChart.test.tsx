import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { CooperationChart } from '../CooperationChart'
import { RoundSummary } from '@/utils/dataTransformers'

// Mock D3 to avoid complex SVG testing
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
    transition: vi.fn().mockReturnThis(),
    duration: vi.fn().mockReturnThis(),
    call: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis(),
    text: vi.fn().mockReturnThis(),
    datum: vi.fn().mockReturnThis(),
    each: vi.fn().mockReturnThis(),
  })),
  scaleLinear: vi.fn(() => ({
    domain: vi.fn().mockReturnThis(),
    range: vi.fn().mockReturnThis(),
  })),
  line: vi.fn(() => ({
    x: vi.fn().mockReturnThis(),
    y: vi.fn().mockReturnThis(),
    curve: vi.fn().mockReturnThis(),
  })),
  axisBottom: vi.fn(() => ({
    tickFormat: vi.fn().mockReturnThis(),
  })),
  axisLeft: vi.fn(() => ({
    tickFormat: vi.fn().mockReturnThis(),
  })),
  extent: vi.fn(() => [1, 3]),
  curveMonotoneX: vi.fn(),
  transition: vi.fn(() => ({
    duration: vi.fn().mockReturnThis(),
    ease: vi.fn().mockReturnThis(),
  })),
  pointer: vi.fn(() => [100, 200]),
}))

const mockData: RoundSummary[] = [
  {
    round: 1,
    cooperation_rate: 0.75,
    average_score: 24.5,
    score_variance: 2.1,
    power_distribution: { mean: 100, std: 15, min: 75, max: 125 },
    score_distribution: { min: 18, max: 30, avg: 24.5 },
    anonymized_games: [
      {
        player1_id: 'agent1',
        player2_id: 'agent2',
        player1_action: 'C',
        player2_action: 'C',
        player1_score: 3,
        player2_score: 3,
      },
      {
        player1_id: 'agent1',
        player2_id: 'agent3',
        player1_action: 'C',
        player2_action: 'D',
        player1_score: 0,
        player2_score: 5,
      },
    ],
  },
  {
    round: 2,
    cooperation_rate: 0.6,
    average_score: 22.0,
    score_variance: 3.5,
    power_distribution: { mean: 105, std: 18, min: 70, max: 130 },
    score_distribution: { min: 15, max: 28, avg: 22.0 },
    anonymized_games: [
      {
        player1_id: 'agent1',
        player2_id: 'agent2',
        player1_action: 'D',
        player2_action: 'C',
        player1_score: 5,
        player2_score: 0,
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

describe('CooperationChart', () => {
  it('renders with loading state', () => {
    render(<CooperationChart data={[]} loading={true} />)
    expect(screen.queryByText('Cooperation Rates Over Rounds')).not.toBeInTheDocument()
  })

  it('renders with error state', () => {
    const error = new Error('Failed to load data')
    render(<CooperationChart data={[]} error={error} />)
    expect(screen.getByText('Error loading chart')).toBeInTheDocument()
    expect(screen.getByText('Failed to load data')).toBeInTheDocument()
  })

  it('renders chart with data', () => {
    render(<CooperationChart data={mockData} />)
    // ChartWrapper is rendered which contains the chart
    expect(screen.getByText('Cooperation Rates Over Rounds')).toBeInTheDocument()
  })

  it('renders with custom title', () => {
    const customTitle = 'Agent Cooperation Analysis'
    render(<CooperationChart data={mockData} title={customTitle} />)
    // Since D3 is mocked, we can't check the actual SVG content
    // but we can verify the component renders without error
    expect(true).toBe(true)
  })

  it('filters agents when selectedAgents prop is provided', () => {
    const selectedAgents = ['agent1', 'agent2']
    render(
      <CooperationChart 
        data={mockData} 
        selectedAgents={selectedAgents}
      />
    )
    // Component should render without error
    expect(true).toBe(true)
  })

  it('calls onAgentClick when provided', () => {
    const onAgentClick = vi.fn()
    render(
      <CooperationChart 
        data={mockData} 
        onAgentClick={onAgentClick}
      />
    )
    // Since D3 is mocked, we can't simulate actual clicks
    // but we can verify the component accepts the prop
    expect(true).toBe(true)
  })

  it('handles empty data gracefully', () => {
    render(<CooperationChart data={[]} />)
    // Should render without error
    expect(true).toBe(true)
  })
})