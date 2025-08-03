import React, { useState } from 'react';
import { XMarkIcon, FunnelIcon } from '@heroicons/react/24/outline';
import { VisualizationFilters, FilterState } from './VisualizationFilters';
import { shouldUseDrawer } from '../../utils/responsive';

interface MobileFilterDrawerProps {
  availableRounds: { min: number; max: number };
  availableAgents: string[];
  onFilterChange: (filters: FilterState) => void;
}

export const MobileFilterDrawer: React.FC<MobileFilterDrawerProps> = ({
  availableRounds,
  availableAgents,
  onFilterChange
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const isMobile = shouldUseDrawer();

  if (!isMobile) {
    // On desktop, just render the filters normally
    return (
      <VisualizationFilters
        availableRounds={availableRounds}
        availableAgents={availableAgents}
        onFilterChange={onFilterChange}
      />
    );
  }

  return (
    <>
      {/* Mobile trigger button */}
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-4 right-4 z-40 flex items-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-full shadow-lg hover:bg-blue-700 transition-colors md:hidden"
      >
        <FunnelIcon className="w-5 h-5" />
        <span>Filters</span>
      </button>

      {/* Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black bg-opacity-50 transition-opacity md:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Drawer */}
      <div
        className={`fixed inset-y-0 right-0 z-50 w-full max-w-sm bg-white shadow-xl transition-transform md:hidden ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold">Visualization Filters</h2>
          <button
            onClick={() => setIsOpen(false)}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <XMarkIcon className="w-5 h-5" />
          </button>
        </div>

        <div className="overflow-y-auto h-full pb-20">
          <VisualizationFilters
            availableRounds={availableRounds}
            availableAgents={availableAgents}
            onFilterChange={(filters) => {
              onFilterChange(filters);
              setIsOpen(false); // Close drawer after applying filters
            }}
            className="border-0 shadow-none"
          />
        </div>
      </div>
    </>
  );
};