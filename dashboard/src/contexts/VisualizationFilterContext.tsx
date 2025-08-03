import React, { createContext, useContext, useState, ReactNode } from 'react';
import { FilterState } from '../components/visualizations/VisualizationFilters';

interface VisualizationFilterContextType {
  filters: FilterState;
  setFilters: (filters: FilterState) => void;
}

export const VisualizationFilterContext = createContext<VisualizationFilterContextType | undefined>(undefined);

export const useVisualizationFilters = () => {
  const context = useContext(VisualizationFilterContext);
  if (!context) {
    throw new Error('useVisualizationFilters must be used within a VisualizationFilterProvider');
  }
  return context;
};

interface VisualizationFilterProviderProps {
  children: ReactNode;
  initialFilters?: FilterState;
}

export const VisualizationFilterProvider: React.FC<VisualizationFilterProviderProps> = ({
  children,
  initialFilters = {
    roundRange: { min: 0, max: 100 },
    selectedAgents: [],
    gameType: 'all'
  }
}) => {
  const [filters, setFilters] = useState<FilterState>(initialFilters);

  return (
    <VisualizationFilterContext.Provider value={{ filters, setFilters }}>
      {children}
    </VisualizationFilterContext.Provider>
  );
};