import React, { useState } from 'react';
import { ArrowDownTrayIcon } from '@heroicons/react/24/outline';
import { exportChart } from '../../utils/chartHelpers';

interface ExportButtonProps {
  svgRef: React.RefObject<SVGSVGElement>;
  filename: string;
  className?: string;
}

export const ExportButton: React.FC<ExportButtonProps> = ({
  svgRef,
  filename,
  className = ''
}) => {
  const [showMenu, setShowMenu] = useState(false);
  const [exporting, setExporting] = useState(false);

  const handleExport = async (format: 'png' | 'svg', resolution: number = 2) => {
    if (!svgRef.current) return;
    
    setExporting(true);
    setShowMenu(false);
    
    try {
      await exportChart(svgRef.current, filename, format, resolution);
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
        className="flex items-center gap-1 px-3 py-1.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
      >
        <ArrowDownTrayIcon className="w-4 h-4" />
        {exporting ? 'Exporting...' : 'Export'}
      </button>

      {showMenu && (
        <div className="absolute right-0 z-10 mt-1 bg-white rounded-md shadow-lg border border-gray-200">
          <div className="py-1">
            <button
              onClick={() => handleExport('png', 2)}
              className="block w-full px-4 py-2 text-sm text-left text-gray-700 hover:bg-gray-100"
            >
              Export as PNG (2x)
            </button>
            <button
              onClick={() => handleExport('png', 4)}
              className="block w-full px-4 py-2 text-sm text-left text-gray-700 hover:bg-gray-100"
            >
              Export as PNG (4x)
            </button>
            <button
              onClick={() => handleExport('svg')}
              className="block w-full px-4 py-2 text-sm text-left text-gray-700 hover:bg-gray-100"
            >
              Export as SVG
            </button>
          </div>
        </div>
      )}
    </div>
  );
};