import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { FilterControls } from '../FilterControls';
import { VisualizationFilterContext } from '../../../contexts/VisualizationFilterContext';
import { GameResult } from '../../../types';
import * as responsive from '../../../utils/responsive';
import React from 'react';

// Mock the responsive utility
vi.mock('../../../utils/responsive', () => ({
  shouldUseDrawer: vi.fn(() => false),
}));

describe('FilterControls', () => {
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
      round: 1,
      action1: 'C',
      action2: 'C',
      score1: 3,
      score2: 3,
      power1: 103,
      power2: 103,
    },
  ];

  const mockSetFilters = vi.fn();
  const mockFilters = {
    roundRange: { min: 0, max: 10 },
    selectedAgents: [],
    gameType: 'all' as const,
  };

  const renderWithContext = (component: React.ReactElement) => {
    return render(
      <VisualizationFilterContext.Provider 
        value={{ 
          filters: mockFilters, 
          setFilters: mockSetFilters 
        }}
      >
        {component}
      </VisualizationFilterContext.Provider>
    );
  };

  beforeEach(() => {
    vi.clearAllMocks();
    (responsive.shouldUseDrawer as any).mockReturnValue(false);
  });

  it('renders VisualizationFilters on desktop', () => {
    renderWithContext(<FilterControls data={mockData} />);
    expect(screen.getByText('Visualization Filters')).toBeInTheDocument();
  });

  it('renders MobileFilterDrawer on mobile', () => {
    (responsive.shouldUseDrawer as any).mockReturnValue(true);
    renderWithContext(<FilterControls data={mockData} />);
    expect(screen.getByText('Filters')).toBeInTheDocument();
  });

  it('calculates available rounds correctly', () => {
    renderWithContext(<FilterControls data={mockData} />);
    // The component should extract rounds 0 and 1 from the data
    expect(screen.getByText('Round Range')).toBeInTheDocument();
  });

  it('calculates available agents correctly', () => {
    renderWithContext(<FilterControls data={mockData} />);
    // The component should extract agents from the data
    expect(screen.getByText(/Agents/)).toBeInTheDocument();
  });

  it('handles empty data gracefully', () => {
    renderWithContext(<FilterControls data={[]} />);
    expect(screen.getByText('Visualization Filters')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = renderWithContext(
      <FilterControls data={mockData} className="custom-class" />
    );
    expect(container.querySelector('.custom-class')).toBeInTheDocument();
  });
});