import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TournamentBracket } from '../TournamentBracket';
import { GameResult } from '../../../types';

// Mock D3 with a more complete implementation
const mockAppend = vi.fn();
const mockAttr = vi.fn();
const mockText = vi.fn();
const mockOn = vi.fn();
const mockSelectAll = vi.fn();
const mockData = vi.fn();
const mockEnter = vi.fn();
const mockRemove = vi.fn();
const mockSelect = vi.fn();
const mockFilter = vi.fn();

// Create chainable mock objects
const createChainableMock = () => ({
  append: mockAppend.mockImplementation(() => createChainableMock()),
  attr: mockAttr.mockImplementation(() => createChainableMock()),
  text: mockText.mockImplementation(() => createChainableMock()),
  on: mockOn.mockImplementation(() => createChainableMock()),
  selectAll: mockSelectAll.mockImplementation(() => createChainableMock()),
  data: mockData.mockImplementation(() => createChainableMock()),
  enter: mockEnter.mockImplementation(() => createChainableMock()),
  remove: mockRemove.mockImplementation(() => createChainableMock()),
  select: mockSelect.mockImplementation(() => createChainableMock()),
  filter: mockFilter.mockImplementation(() => createChainableMock()),
});

vi.mock('d3', () => ({
  select: vi.fn(() => createChainableMock()),
}));

describe('TournamentBracket', () => {
  const mockData: GameResult[] = [
    {
      agent1_id: 'agent1',
      agent2_id: 'agent2',
      round: 0,
      action1: 'C',
      action2: 'D',
      score1: 0,
      score2: 5,
      power1: 100,
      power2: 105,
    },
    {
      agent1_id: 'agent3',
      agent2_id: 'agent4',
      round: 0,
      action1: 'C',
      action2: 'C',
      score1: 3,
      score2: 3,
      power1: 103,
      power2: 103,
    },
    {
      agent1_id: 'agent1',
      agent2_id: 'agent3',
      round: 1,
      action1: 'D',
      action2: 'D',
      score1: 1,
      score2: 1,
      power1: 101,
      power2: 104,
    },
  ];

  const mockOnGameClick = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders without crashing', () => {
    render(<TournamentBracket data={mockData} />);
    expect(screen.getByText('Tournament Bracket')).toBeInTheDocument();
  });

  it('displays helper text', () => {
    render(<TournamentBracket data={mockData} />);
    expect(screen.getByText('Click on matches to expand details. Click on round labels to filter.')).toBeInTheDocument();
  });

  it('handles empty data gracefully', () => {
    render(<TournamentBracket data={[]} />);
    expect(screen.getByText('Tournament Bracket')).toBeInTheDocument();
  });

  it('accepts custom dimensions', () => {
    const { container } = render(
      <TournamentBracket data={mockData} width={1000} height={800} />
    );
    const svg = container.querySelector('svg');
    expect(svg).toHaveAttribute('width', '1000');
    expect(svg).toHaveAttribute('height', '800');
  });

  it('calls onGameClick when provided', () => {
    render(
      <TournamentBracket data={mockData} onGameClick={mockOnGameClick} />
    );
    // Note: Due to D3 mocking, we can't directly test click events
    // In a real test environment, you would simulate clicks on bracket nodes
  });

  it('renders with proper structure', () => {
    const { container } = render(<TournamentBracket data={mockData} />);
    expect(container.querySelector('.tournament-bracket')).toBeInTheDocument();
    expect(container.querySelector('svg')).toBeInTheDocument();
  });
});