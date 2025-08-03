import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { VisualizationFilterProvider } from '../contexts/VisualizationFilterContext';
import { VisualizationDashboard } from '../components/visualizations/VisualizationDashboard';
import { ErrorBoundary } from '../components/error/ErrorBoundary';

export const ExperimentVisualization: React.FC = () => {
  const { experimentId } = useParams<{ experimentId: string }>();
  const navigate = useNavigate();

  if (!experimentId) {
    return (
      <div className="text-center py-12">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          No experiment selected
        </h2>
        <button
          onClick={() => navigate('/')}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          Back to Dashboard
        </button>
      </div>
    );
  }

  return (
    <div>
      {/* Page Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              Experiment Visualizations
            </h1>
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
              Analyze tournament results with interactive charts
            </p>
          </div>
          <button
            onClick={() => navigate('/')}
            className="px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
          >
            Back to Dashboard
          </button>
        </div>
      </div>

      {/* Visualization Dashboard with Filter Context */}
      <ErrorBoundary>
        <VisualizationFilterProvider>
          <VisualizationDashboard experimentId={experimentId} />
        </VisualizationFilterProvider>
      </ErrorBoundary>
    </div>
  );
};