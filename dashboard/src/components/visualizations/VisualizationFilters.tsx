import React, { useState, useCallback, useMemo } from 'react';
import { ChevronDownIcon, FunnelIcon } from '@heroicons/react/24/outline';

export interface FilterState {
  roundRange: { min: number; max: number };
  selectedAgents: string[];
  gameType: 'all' | 'cooperate' | 'defect' | 'mixed';
}

interface VisualizationFiltersProps {
  availableRounds: { min: number; max: number };
  availableAgents: string[];
  onFilterChange: (filters: FilterState) => void;
  className?: string;
}

export const VisualizationFilters: React.FC<VisualizationFiltersProps> = ({
  availableRounds,
  availableAgents,
  onFilterChange,
  className = ''
}) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [roundMin, setRoundMin] = useState(availableRounds.min);
  const [roundMax, setRoundMax] = useState(availableRounds.max);
  const [selectedAgents, setSelectedAgents] = useState<string[]>(availableAgents);
  const [gameType, setGameType] = useState<FilterState['gameType']>('all');
  const [agentSearchTerm, setAgentSearchTerm] = useState('');
  const [showAgentDropdown, setShowAgentDropdown] = useState(false);

  const filteredAgents = useMemo(() => {
    if (!agentSearchTerm) return availableAgents;
    return availableAgents.filter(agent => 
      agent.toLowerCase().includes(agentSearchTerm.toLowerCase())
    );
  }, [availableAgents, agentSearchTerm]);

  const applyFilters = useCallback(() => {
    onFilterChange({
      roundRange: { min: roundMin, max: roundMax },
      selectedAgents,
      gameType
    });
  }, [roundMin, roundMax, selectedAgents, gameType, onFilterChange]);

  const resetFilters = useCallback(() => {
    setRoundMin(availableRounds.min);
    setRoundMax(availableRounds.max);
    setSelectedAgents(availableAgents);
    setGameType('all');
    onFilterChange({
      roundRange: availableRounds,
      selectedAgents: availableAgents,
      gameType: 'all'
    });
  }, [availableRounds, availableAgents, onFilterChange]);

  const toggleAgent = (agent: string) => {
    setSelectedAgents(prev => {
      if (prev.includes(agent)) {
        return prev.filter(a => a !== agent);
      } else {
        return [...prev, agent];
      }
    });
  };

  const selectAllAgents = () => {
    setSelectedAgents(availableAgents);
  };

  const deselectAllAgents = () => {
    setSelectedAgents([]);
  };

  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 ${className}`}>
      <button
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-gray-50 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <FunnelIcon className="w-5 h-5 text-gray-500" />
          <span className="font-medium text-gray-900">Visualization Filters</span>
        </div>
        <ChevronDownIcon
          className={`w-5 h-5 text-gray-500 transition-transform ${
            isExpanded ? 'transform rotate-180' : ''
          }`}
        />
      </button>

      {isExpanded && (
        <div className="p-4 border-t border-gray-200 space-y-4">
          {/* Round Range Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Round Range
            </label>
            <div className="flex items-center gap-3">
              <input
                type="number"
                min={availableRounds.min}
                max={availableRounds.max}
                value={roundMin}
                onChange={(e) => setRoundMin(Number(e.target.value))}
                className="w-20 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <span className="text-gray-500">to</span>
              <input
                type="number"
                min={availableRounds.min}
                max={availableRounds.max}
                value={roundMax}
                onChange={(e) => setRoundMax(Number(e.target.value))}
                className="w-20 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          {/* Agent Multi-Select */}
          <div className="relative">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Agents ({selectedAgents.length} of {availableAgents.length} selected)
            </label>
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowAgentDropdown(!showAgentDropdown)}
                className="w-full px-3 py-2 text-left bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <span className="block truncate">
                  {selectedAgents.length === availableAgents.length
                    ? 'All agents selected'
                    : selectedAgents.length === 0
                    ? 'No agents selected'
                    : `${selectedAgents.length} agents selected`}
                </span>
                <span className="absolute inset-y-0 right-0 flex items-center pr-2">
                  <ChevronDownIcon className="w-5 h-5 text-gray-400" />
                </span>
              </button>

              {showAgentDropdown && (
                <div className="absolute z-10 w-full mt-1 bg-white rounded-md shadow-lg">
                  <div className="p-2 border-b border-gray-200">
                    <input
                      type="text"
                      placeholder="Search agents..."
                      value={agentSearchTerm}
                      onChange={(e) => setAgentSearchTerm(e.target.value)}
                      className="w-full px-3 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                    />
                  </div>
                  <div className="p-2 border-b border-gray-200 flex gap-2">
                    <button
                      onClick={selectAllAgents}
                      className="text-xs text-blue-600 hover:text-blue-800"
                    >
                      Select All
                    </button>
                    <button
                      onClick={deselectAllAgents}
                      className="text-xs text-blue-600 hover:text-blue-800"
                    >
                      Deselect All
                    </button>
                  </div>
                  <div className="max-h-48 overflow-y-auto">
                    {filteredAgents.map(agent => (
                      <label
                        key={agent}
                        className="flex items-center px-3 py-2 hover:bg-gray-50 cursor-pointer"
                      >
                        <input
                          type="checkbox"
                          checked={selectedAgents.includes(agent)}
                          onChange={() => toggleAgent(agent)}
                          className="mr-3 h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                        />
                        <span className="text-sm text-gray-700">{agent}</span>
                      </label>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Game Type Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Game Type
            </label>
            <select
              value={gameType}
              onChange={(e) => setGameType(e.target.value as FilterState['gameType'])}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Games</option>
              <option value="cooperate">Mutual Cooperation (C-C)</option>
              <option value="defect">Mutual Defection (D-D)</option>
              <option value="mixed">Mixed (C-D or D-C)</option>
            </select>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2 pt-2">
            <button
              onClick={applyFilters}
              className="flex-1 px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              Apply Filters
            </button>
            <button
              onClick={resetFilters}
              className="flex-1 px-4 py-2 bg-gray-100 text-gray-700 text-sm font-medium rounded-md hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-gray-500"
            >
              Reset
            </button>
          </div>
        </div>
      )}
    </div>
  );
};