import React, { useState } from 'react';
import { ArrowDownTrayIcon } from '@heroicons/react/24/outline';
import { exportChart } from '../../utils/chartHelpers';

interface BatchExportProps {
  charts: Array<{
    ref: React.RefObject<SVGSVGElement>;
    filename: string;
    title: string;
  }>;
  className?: string;
}

export const BatchExport: React.FC<BatchExportProps> = ({
  charts,
  className = ''
}) => {
  const [exporting, setExporting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [showMenu, setShowMenu] = useState(false);

  const handleBatchExport = async (format: 'png' | 'svg', resolution: number = 2) => {
    setExporting(true);
    setShowMenu(false);
    setProgress(0);
    
    const validCharts = charts.filter(chart => chart.ref.current);
    
    for (let i = 0; i < validCharts.length; i++) {
      const chart = validCharts[i];
      if (chart.ref.current) {
        try {
          await exportChart(chart.ref.current, chart.filename, format, resolution);
          setProgress(((i + 1) / validCharts.length) * 100);
          // Small delay between exports to prevent browser overwhelm
          await new Promise(resolve => setTimeout(resolve, 100));
        } catch (error) {
          console.error(`Failed to export ${chart.filename}:`, error);
        }
      }
    }
    
    setExporting(false);
    setProgress(0);
  };

  return (
    <div className={`relative ${className}`}>
      <button
        onClick={() => setShowMenu(!showMenu)}
        disabled={exporting}
        className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
      >
        <ArrowDownTrayIcon className="w-5 h-5" />
        {exporting ? `Exporting... ${Math.round(progress)}%` : 'Export All Charts'}
      </button>

      {showMenu && (
        <div className="absolute right-0 z-10 mt-1 bg-white rounded-md shadow-lg border border-gray-200">
          <div className="py-1">
            <div className="px-4 py-2 text-xs font-medium text-gray-500 uppercase tracking-wider">
              Export {charts.length} charts
            </div>
            <button
              onClick={() => handleBatchExport('png', 2)}
              className="block w-full px-4 py-2 text-sm text-left text-gray-700 hover:bg-gray-100"
            >
              All as PNG (2x resolution)
            </button>
            <button
              onClick={() => handleBatchExport('png', 4)}
              className="block w-full px-4 py-2 text-sm text-left text-gray-700 hover:bg-gray-100"
            >
              All as PNG (4x resolution)
            </button>
            <button
              onClick={() => handleBatchExport('svg')}
              className="block w-full px-4 py-2 text-sm text-left text-gray-700 hover:bg-gray-100"
            >
              All as SVG
            </button>
          </div>
        </div>
      )}

      {exporting && (
        <div className="absolute bottom-full mb-2 left-0 right-0 bg-white rounded-md shadow-sm border border-gray-200 p-2">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
};