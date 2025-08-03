import React, { useState, useEffect, useContext } from 'react';
import { VisualizationFilterContext } from '../../contexts/VisualizationFilterContext';
import { FilterControls } from './FilterControls';
import { TournamentBracket } from './TournamentBracket';
import { CooperationChart } from './CooperationChart';
import { PowerDistributionChart } from './PowerDistributionChart';
import { ScoreProgressionChart } from './ScoreProgressionChart';
import { MatchupHeatMap } from './MatchupHeatMap';
import { DataExport } from './DataExport';
import { ErrorBoundary } from '../error/ErrorBoundary';
import { Skeleton } from '../loading/Skeleton';
import { GameResult } from '../../types';
import { api } from '../../services/api';
import { useToast } from '../../hooks/useToast';
import { applyVisualizationFilters } from '../../utils/dataTransformers';

interface VisualizationDashboardProps {
  experimentId: string;
}

export const VisualizationDashboard: React.FC<VisualizationDashboardProps> = ({ 
  experimentId 
}) => {
  const [data, setData] = useState<GameResult[]>([]);
  const [loading, setLoading] = useState(true);
  const { filters } = useContext(VisualizationFilterContext);
  const { error } = useToast();

  useEffect(() => {
    fetchGameResults();
  }, [experimentId]);

  const fetchGameResults = async () => {
    try {
      setLoading(true);
      const results = await api.getGameResults(experimentId);
      setData(results);
    } catch (err) {
      error('Failed to load game results', 'Please try again later');
      console.error('Error fetching game results:', err);
    } finally {
      setLoading(false);
    }
  };

  // Apply filters to data
  const filteredData = applyVisualizationFilters(data, filters);

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton lines={3} />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} height={300} />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Filter Controls */}
      <FilterControls data={data} className="mb-6" />

      {/* Export Options */}
      <div className="flex justify-end">
        <DataExport
          experimentId={experimentId}
          data={filteredData}
          filters={filters}
        />
      </div>

      {/* Visualizations Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Tournament Bracket */}
        <div className="lg:col-span-2">
          <ErrorBoundary>
            <TournamentBracket 
              data={filteredData}
              onGameClick={(game) => {
                console.log('Game clicked:', game);
              }}
            />
          </ErrorBoundary>
        </div>

        {/* Cooperation Chart */}
        <ErrorBoundary>
          <CooperationChart 
            data={filteredData}
            selectedAgents={filters.selectedAgents}
            onAgentClick={(agent) => {
              console.log('Agent clicked:', agent);
            }}
          />
        </ErrorBoundary>

        {/* Power Distribution */}
        <ErrorBoundary>
          <PowerDistributionChart 
            data={filteredData}
            selectedRound={filters.roundRange.max}
            showEvolution={true}
          />
        </ErrorBoundary>

        {/* Score Progression */}
        <ErrorBoundary>
          <ScoreProgressionChart 
            data={filteredData}
            selectedAgents={filters.selectedAgents}
            chartType="line"
            showAnimation={true}
          />
        </ErrorBoundary>

        {/* Matchup Heat Map */}
        <ErrorBoundary>
          <MatchupHeatMap 
            data={filteredData}
            colorScale="diverging"
            onCellClick={(agent1, agent2) => {
              console.log('Cell clicked:', agent1, agent2);
            }}
          />
        </ErrorBoundary>
      </div>

      {/* Empty State */}
      {filteredData.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500">
            No data matches the current filters. Try adjusting your selection.
          </p>
        </div>
      )}
    </div>
  );
};