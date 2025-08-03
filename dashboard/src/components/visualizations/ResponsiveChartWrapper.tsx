import React, { useEffect, useState, useRef, ReactNode } from 'react';
import { ChartWrapper } from './ChartWrapper';
import { ExportButton } from './ExportButton';
import { getChartDimensions, onResize, isTouchDevice } from '../../utils/responsive';

interface ResponsiveChartWrapperProps {
  title: string;
  children: (dimensions: { width: number; height: number }) => ReactNode;
  svgRef?: React.RefObject<SVGSVGElement>;
  exportFilename?: string;
  className?: string;
  minHeight?: number;
}

export const ResponsiveChartWrapper: React.FC<ResponsiveChartWrapperProps> = ({
  title,
  children,
  svgRef,
  exportFilename,
  className = '',
  minHeight = 240
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState(getChartDimensions());
  const [isTouch, setIsTouch] = useState(false);

  useEffect(() => {
    // Set initial touch state
    setIsTouch(isTouchDevice());

    // Update dimensions based on container width
    const updateDimensions = () => {
      if (containerRef.current) {
        const containerWidth = containerRef.current.offsetWidth;
        const newDimensions = getChartDimensions(containerWidth);
        setDimensions({
          width: Math.min(newDimensions.width, containerWidth),
          height: Math.max(newDimensions.height, minHeight)
        });
      }
    };

    // Initial dimension calculation
    updateDimensions();

    // Set up resize listener
    const cleanup = onResize(updateDimensions);

    return cleanup;
  }, [minHeight]);

  return (
    <ChartWrapper
      title={title}
      className={`responsive-chart ${className}`}
      action={
        svgRef && exportFilename && (
          <ExportButton svgRef={svgRef} filename={exportFilename} />
        )
      }
    >
      <div 
        ref={containerRef}
        className={`relative ${isTouch ? 'touch-device' : ''}`}
        style={{ minHeight: `${minHeight}px` }}
      >
        {children(dimensions)}
      </div>
      {isTouch && (
        <div className="mt-2 text-xs text-gray-500 text-center">
          Pinch to zoom • Drag to pan
        </div>
      )}
    </ChartWrapper>
  );
};