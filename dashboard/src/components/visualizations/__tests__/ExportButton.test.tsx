import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ExportButton } from '../ExportButton';
import * as chartHelpers from '../../../utils/chartHelpers';

// Mock the export function
vi.mock('../../../utils/chartHelpers', () => ({
  exportChart: vi.fn().mockResolvedValue(undefined),
}));

describe('ExportButton', () => {
  const mockSvgRef = {
    current: document.createElementNS('http://www.w3.org/2000/svg', 'svg'),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders without crashing', () => {
    render(<ExportButton svgRef={mockSvgRef} filename="test-chart" />);
    expect(screen.getByText('Export')).toBeInTheDocument();
  });

  it('shows export menu when clicked', () => {
    render(<ExportButton svgRef={mockSvgRef} filename="test-chart" />);
    
    const button = screen.getByText('Export');
    fireEvent.click(button);
    
    expect(screen.getByText('Export as PNG (2x)')).toBeInTheDocument();
    expect(screen.getByText('Export as PNG (4x)')).toBeInTheDocument();
    expect(screen.getByText('Export as SVG')).toBeInTheDocument();
  });

  it('hides menu when clicking outside', () => {
    render(<ExportButton svgRef={mockSvgRef} filename="test-chart" />);
    
    const button = screen.getByText('Export');
    fireEvent.click(button);
    expect(screen.getByText('Export as PNG (2x)')).toBeInTheDocument();
    
    fireEvent.click(button);
    expect(screen.queryByText('Export as PNG (2x)')).not.toBeInTheDocument();
  });

  it('exports as PNG with 2x resolution', async () => {
    render(<ExportButton svgRef={mockSvgRef} filename="test-chart" />);
    
    const button = screen.getByText('Export');
    fireEvent.click(button);
    
    const pngButton = screen.getByText('Export as PNG (2x)');
    fireEvent.click(pngButton);
    
    await waitFor(() => {
      expect(chartHelpers.exportChart).toHaveBeenCalledWith(
        mockSvgRef.current,
        'test-chart',
        'png',
        2
      );
    });
  });

  it('exports as PNG with 4x resolution', async () => {
    render(<ExportButton svgRef={mockSvgRef} filename="test-chart" />);
    
    const button = screen.getByText('Export');
    fireEvent.click(button);
    
    const pngButton = screen.getByText('Export as PNG (4x)');
    fireEvent.click(pngButton);
    
    await waitFor(() => {
      expect(chartHelpers.exportChart).toHaveBeenCalledWith(
        mockSvgRef.current,
        'test-chart',
        'png',
        4
      );
    });
  });

  it('exports as SVG', async () => {
    render(<ExportButton svgRef={mockSvgRef} filename="test-chart" />);
    
    const button = screen.getByText('Export');
    fireEvent.click(button);
    
    const svgButton = screen.getByText('Export as SVG');
    fireEvent.click(svgButton);
    
    await waitFor(() => {
      expect(chartHelpers.exportChart).toHaveBeenCalledWith(
        mockSvgRef.current,
        'test-chart',
        'svg',
        2
      );
    });
  });

  it('shows exporting state', async () => {
    render(<ExportButton svgRef={mockSvgRef} filename="test-chart" />);
    
    const button = screen.getByText('Export');
    fireEvent.click(button);
    
    const pngButton = screen.getByText('Export as PNG (2x)');
    fireEvent.click(pngButton);
    
    expect(screen.getByText('Exporting...')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(screen.getByText('Export')).toBeInTheDocument();
    });
  });

  it('handles missing SVG ref gracefully', async () => {
    const emptyRef = { current: null };
    render(<ExportButton svgRef={emptyRef} filename="test-chart" />);
    
    const button = screen.getByText('Export');
    fireEvent.click(button);
    
    const pngButton = screen.getByText('Export as PNG (2x)');
    fireEvent.click(pngButton);
    
    expect(chartHelpers.exportChart).not.toHaveBeenCalled();
  });

  it('applies custom className', () => {
    const { container } = render(
      <ExportButton svgRef={mockSvgRef} filename="test-chart" className="custom-class" />
    );
    
    expect(container.querySelector('.custom-class')).toBeInTheDocument();
  });
});