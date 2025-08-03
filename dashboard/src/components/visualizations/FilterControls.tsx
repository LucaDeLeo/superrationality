import React, { useContext } from 'react';
import { VisualizationFilterContext } from '../../contexts/VisualizationFilterContext';
import { VisualizationFilters, FilterState } from './VisualizationFilters';
import { MobileFilterDrawer } from './MobileFilterDrawer';
import { shouldUseDrawer } from '../../utils/responsive';
import { GameResult } from '../../types';

interface FilterControlsProps {
  data: GameResult[];
  className?: string;
}

export const FilterControls: React.FC<FilterControlsProps> = ({ data, className = '' }) => {
  const { setFilters } = useContext(VisualizationFilterContext);
  const isMobile = shouldUseDrawer();

  // Calculate available rounds and agents from data
  const availableRounds = React.useMemo(() => {
    if (!data || data.length === 0) {
      return { min: 0, max: 0 };
    }
    const rounds = data.map(game => game.round || 0);
    return {
      min: Math.min(...rounds),
      max: Math.max(...rounds)
    };
  }, [data]);

  const availableAgents = React.useMemo(() => {
    if (!data || data.length === 0) {
      return [];
    }
    const agentSet = new Set<string>();
    data.forEach(game => {
      agentSet.add(game.agent1_id);
      agentSet.add(game.agent2_id);
    });
    return Array.from(agentSet).sort();
  }, [data]);

  const handleFilterChange = (newFilters: FilterState) => {
    setFilters(newFilters);
  };

  // Use mobile drawer on small screens
  if (isMobile) {
    return (
      <MobileFilterDrawer
        availableRounds={availableRounds}
        availableAgents={availableAgents}
        onFilterChange={handleFilterChange}
      />
    );
  }

  // Use regular filters on desktop
  return (
    <VisualizationFilters
      availableRounds={availableRounds}
      availableAgents={availableAgents}
      onFilterChange={handleFilterChange}
      className={className}
    />
  );
};