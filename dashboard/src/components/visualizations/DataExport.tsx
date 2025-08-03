import React, { useState } from 'react';
import { ArrowDownTrayIcon } from '@heroicons/react/24/outline';
import { GameResult } from '../../types';
import { FilterState } from './VisualizationFilters';

interface DataExportProps {
  experimentId: string;
  data: GameResult[];
  filters: FilterState;
  className?: string;
}

export const DataExport: React.FC<DataExportProps> = ({
  experimentId,
  data,
  filters,
  className = ''
}) => {
  const [showMenu, setShowMenu] = useState(false);
  const [exporting, setExporting] = useState(false);

  const exportAsJSON = () => {
    const exportData = {
      experimentId,
      filters,
      data,
      exportDate: new Date().toISOString(),
      totalGames: data.length
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `experiment-${experimentId}-filtered-data.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const exportAsCSV = () => {
    const headers = ['Round', 'Agent1', 'Agent2', 'Agent1 Choice', 'Agent2 Choice', 'Agent1 Score', 'Agent2 Score', 'Outcome'];
    const rows = data.map(game => [
      game.round_number || '',
      game.agent1_id,
      game.agent2_id,
      game.agent1_choice,
      game.agent2_choice,
      game.agent1_score,
      game.agent2_score,
      game.outcome
    ]);

    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `experiment-${experimentId}-filtered-data.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const exportSummary = () => {
    const summary = {
      experimentId,
      filters,
      totalGames: data.length,
      cooperationRate: data.filter(g => g.agent1_choice === 'COOPERATE' || g.agent2_choice === 'COOPERATE').length / (data.length * 2),
      mutualCooperationRate: data.filter(g => g.agent1_choice === 'COOPERATE' && g.agent2_choice === 'COOPERATE').length / data.length,
      mutualDefectionRate: data.filter(g => g.agent1_choice === 'DEFECT' && g.agent2_choice === 'DEFECT').length / data.length,
      exportDate: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(summary, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `experiment-${experimentId}-summary.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleExport = async (type: 'json' | 'csv' | 'summary') => {
    setExporting(true);
    setShowMenu(false);
    
    try {
      switch (type) {
        case 'json':
          exportAsJSON();
          break;
        case 'csv':
          exportAsCSV();
          break;
        case 'summary':
          exportSummary();
          break;
      }
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className={`relative ${className}`}>
      <button
        onClick={() => setShowMenu(!showMenu)}
        disabled={exporting}
        className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
      >
        <ArrowDownTrayIcon className="w-5 h-5" />
        {exporting ? 'Exporting...' : 'Export Data'}
      </button>

      {showMenu && (
        <div className="absolute right-0 z-10 mt-1 bg-white rounded-md shadow-lg border border-gray-200">
          <div className="py-1">
            <div className="px-4 py-2 text-xs font-medium text-gray-500 uppercase tracking-wider">
              Export {data.length} filtered games
            </div>
            <button
              onClick={() => handleExport('json')}
              className="block w-full px-4 py-2 text-sm text-left text-gray-700 hover:bg-gray-100"
            >
              Export as JSON
            </button>
            <button
              onClick={() => handleExport('csv')}
              className="block w-full px-4 py-2 text-sm text-left text-gray-700 hover:bg-gray-100"
            >
              Export as CSV
            </button>
            <button
              onClick={() => handleExport('summary')}
              className="block w-full px-4 py-2 text-sm text-left text-gray-700 hover:bg-gray-100"
            >
              Export Summary
            </button>
          </div>
        </div>
      )}
    </div>
  );
};