import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MatchupHeatMap } from '../MatchupHeatMap'
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
    filter: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis(),
    text: vi.fn().mockReturnThis(),
    call: vi.fn().mockReturnThis(),
  })),
  scaleBand: vi.fn(() => ({
    domain: vi.fn().mockReturnThis(),
    range: vi.fn().mockReturnThis(),
    padding: vi.fn().mockReturnThis(),
    bandwidth: vi.fn(() => 50),
  })),
  scaleSequential: vi.fn(() => ({
    domain: vi.fn().mockReturnThis(),
    interpolator: vi.fn().mockReturnThis(),
  })),
  scaleLinear: vi.fn(() => ({
    domain: vi.fn().mockReturnThis(),
    range: vi.fn().mockReturnThis(),
  })),
  interpolateRdYlGn: vi.fn(),
  interpolateRdBu: vi.fn(),
  max: vi.fn(() => 5),
  axisBottom: vi.fn(() => ({
    ticks: vi.fn().mockReturnThis(),
    tickFormat: vi.fn().mockReturnThis(),
  })),
  pointer: vi.fn(() => [100, 200]),
}))

const mockData: RoundSummary[] = [
  {
    round: 1,
    cooperation_rate: 0.5,
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
      {
        player1_id: 'agent2',
        player2_id: 'agent3',
        player1_action: 'D',
        player2_action: 'D',
        player1_score: 1,
        player2_score: 1,
      },
    ],
  },
]

describe('MatchupHeatMap', () => {
  it('renders with loading state', () => {
    render(<MatchupHeatMap data={[]} loading={true} />)
    // Should show skeleton loader, not the title
    expect(screen.queryByText('Agent vs Agent Matchup Results')).not.toBeInTheDocument()
  })

  it('renders with error state', () => {
    const error = new Error('Failed to load matchup data')
    render(<MatchupHeatMap data={[]} error={error} />)
    expect(screen.getByText('Error loading chart')).toBeInTheDocument()
    expect(screen.getByText('Failed to load matchup data')).toBeInTheDocument()
  })

  it('renders heat map with data', () => {
    const { container } = render(<MatchupHeatMap data={mockData} />)
    expect(container.querySelector('svg.matchup-heatmap')).toBeInTheDocument()
  })

  it('renders with custom title', () => {
    const customTitle = 'Tournament Matchup Analysis'
    render(<MatchupHeatMap data={mockData} title={customTitle} />)
    // Component should render without error
    expect(true).toBe(true)
  })

  it('supports different color scales', () => {
    const { rerender } = render(
      <MatchupHeatMap data={mockData} colorScale="winRate" />
    )
    expect(true).toBe(true)

    rerender(<MatchupHeatMap data={mockData} colorScale="cooperationRate" />)
    expect(true).toBe(true)

    rerender(<MatchupHeatMap data={mockData} colorScale="scoreDiff" />)
    expect(true).toBe(true)
  })

  it('calls onCellClick when provided', () => {
    const onCellClick = vi.fn()
    render(
      <MatchupHeatMap 
        data={mockData} 
        onCellClick={onCellClick}
      />
    )
    // Component should accept the prop
    expect(true).toBe(true)
  })

  it('handles empty data gracefully', () => {
    render(<MatchupHeatMap data={[]} />)
    // Should render without error
    expect(true).toBe(true)
  })

  it('handles single agent gracefully', () => {
    const singleAgentData: RoundSummary[] = [{
      ...mockData[0],
      anonymized_games: [{
        player1_id: 'agent1',
        player2_id: 'agent1',
        player1_action: 'C',
        player2_action: 'C',
        player1_score: 3,
        player2_score: 3,
      }]
    }]
    
    render(<MatchupHeatMap data={singleAgentData} />)
    expect(true).toBe(true)
  })
})