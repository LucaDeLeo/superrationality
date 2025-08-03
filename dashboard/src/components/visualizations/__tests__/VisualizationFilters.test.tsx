import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { VisualizationFilters, FilterState } from '../VisualizationFilters';

describe('VisualizationFilters', () => {
  const mockOnFilterChange = vi.fn();
  const defaultProps = {
    availableRounds: { min: 0, max: 10 },
    availableAgents: ['agent1', 'agent2', 'agent3', 'agent4'],
    onFilterChange: mockOnFilterChange,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders without crashing', () => {
    render(<VisualizationFilters {...defaultProps} />);
    expect(screen.getByText('Visualization Filters')).toBeInTheDocument();
  });

  it('toggles expansion when header is clicked', () => {
    render(<VisualizationFilters {...defaultProps} />);
    
    const header = screen.getByText('Visualization Filters').closest('button');
    expect(screen.getByText('Round Range')).toBeInTheDocument();
    
    fireEvent.click(header!);
    expect(screen.queryByText('Round Range')).not.toBeInTheDocument();
    
    fireEvent.click(header!);
    expect(screen.getByText('Round Range')).toBeInTheDocument();
  });

  it('updates round range inputs', () => {
    render(<VisualizationFilters {...defaultProps} />);
    
    const minInput = screen.getByDisplayValue('0');
    const maxInput = screen.getByDisplayValue('10');
    
    fireEvent.change(minInput, { target: { value: '2' } });
    fireEvent.change(maxInput, { target: { value: '8' } });
    
    expect(minInput).toHaveValue(2);
    expect(maxInput).toHaveValue(8);
  });

  it('applies filters when Apply button is clicked', () => {
    render(<VisualizationFilters {...defaultProps} />);
    
    const minInput = screen.getByDisplayValue('0');
    fireEvent.change(minInput, { target: { value: '2' } });
    
    const applyButton = screen.getByText('Apply Filters');
    fireEvent.click(applyButton);
    
    expect(mockOnFilterChange).toHaveBeenCalledWith({
      roundRange: { min: 2, max: 10 },
      selectedAgents: defaultProps.availableAgents,
      gameType: 'all',
    });
  });

  it('resets filters when Reset button is clicked', () => {
    render(<VisualizationFilters {...defaultProps} />);
    
    const minInput = screen.getByDisplayValue('0');
    fireEvent.change(minInput, { target: { value: '5' } });
    
    const resetButton = screen.getByText('Reset');
    fireEvent.click(resetButton);
    
    expect(minInput).toHaveValue(0);
    expect(mockOnFilterChange).toHaveBeenCalledWith({
      roundRange: { min: 0, max: 10 },
      selectedAgents: defaultProps.availableAgents,
      gameType: 'all',
    });
  });

  it('toggles agent selection dropdown', () => {
    render(<VisualizationFilters {...defaultProps} />);
    
    const agentButton = screen.getByText('All agents selected').closest('button');
    expect(screen.queryByText('Search agents...')).not.toBeInTheDocument();
    
    fireEvent.click(agentButton!);
    expect(screen.getByPlaceholderText('Search agents...')).toBeInTheDocument();
  });

  it('filters agents based on search term', async () => {
    render(<VisualizationFilters {...defaultProps} />);
    
    const agentButton = screen.getByText('All agents selected').closest('button');
    fireEvent.click(agentButton!);
    
    const searchInput = screen.getByPlaceholderText('Search agents...');
    fireEvent.change(searchInput, { target: { value: 'agent1' } });
    
    await waitFor(() => {
      expect(screen.getByLabelText('agent1')).toBeInTheDocument();
      expect(screen.queryByLabelText('agent2')).not.toBeInTheDocument();
    });
  });

  it('selects and deselects all agents', () => {
    render(<VisualizationFilters {...defaultProps} />);
    
    const agentButton = screen.getByText('All agents selected').closest('button');
    fireEvent.click(agentButton!);
    
    const deselectAllButton = screen.getByText('Deselect All');
    fireEvent.click(deselectAllButton);
    
    const applyButton = screen.getByText('Apply Filters');
    fireEvent.click(applyButton);
    
    expect(mockOnFilterChange).toHaveBeenCalledWith(
      expect.objectContaining({
        selectedAgents: [],
      })
    );
  });

  it('changes game type filter', () => {
    render(<VisualizationFilters {...defaultProps} />);
    
    const gameTypeSelect = screen.getByRole('combobox');
    fireEvent.change(gameTypeSelect, { target: { value: 'cooperate' } });
    
    const applyButton = screen.getByText('Apply Filters');
    fireEvent.click(applyButton);
    
    expect(mockOnFilterChange).toHaveBeenCalledWith(
      expect.objectContaining({
        gameType: 'cooperate',
      })
    );
  });

  it('applies custom className', () => {
    const { container } = render(
      <VisualizationFilters {...defaultProps} className="custom-class" />
    );
    
    expect(container.querySelector('.custom-class')).toBeInTheDocument();
  });
});