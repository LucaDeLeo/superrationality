export const chartTheme = {
  colors: {
    primary: '#3b82f6', // blue-500
    secondary: '#8b5cf6', // violet-500
    tertiary: '#10b981', // emerald-500
    quaternary: '#f59e0b', // amber-500
    danger: '#ef4444', // red-500
    warning: '#f97316', // orange-500
    success: '#22c55e', // green-500
    neutral: '#6b7280', // gray-500
    
    // Extended palette for multiple agents
    palette: [
      '#3b82f6', // blue
      '#8b5cf6', // violet
      '#10b981', // emerald
      '#f59e0b', // amber
      '#ec4899', // pink
      '#14b8a6', // teal
      '#f97316', // orange
      '#06b6d4', // cyan
      '#a855f7', // purple
      '#84cc16', // lime
      '#0ea5e9', // sky
      '#eab308', // yellow
    ],
    
    // Heatmap colors
    heatmap: {
      defect: '#ef4444', // red-500
      neutral: '#fbbf24', // amber-400
      cooperate: '#22c55e', // green-500
    },
    
    // Background and grid
    background: '#ffffff',
    grid: '#e5e7eb', // gray-200
    text: '#374151', // gray-700
    textLight: '#6b7280', // gray-500
  },
  
  fonts: {
    family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    sizes: {
      title: 16,
      axis: 12,
      label: 11,
      tooltip: 12,
    },
  },
  
  margins: {
    top: 40,
    right: 80,
    bottom: 60,
    left: 80,
  },
  
  animation: {
    duration: 300,
    easing: 'easeInOutQuad',
  },
  
  gridLines: {
    stroke: '#e5e7eb',
    strokeWidth: 1,
    strokeDasharray: '3,3',
  },
  
  axis: {
    stroke: '#6b7280',
    strokeWidth: 1,
    tick: {
      size: 5,
      padding: 8,
    },
  },
  
  tooltip: {
    background: '#1f2937', // gray-800
    color: '#ffffff',
    padding: 8,
    borderRadius: 4,
    fontSize: 12,
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
  },
}

// Helper to get color for agent by index
export const getAgentColor = (index: number): string => {
  const { palette } = chartTheme.colors
  return palette[index % palette.length]
}

// Helper to get cooperation color scale
export const getCooperationColor = (cooperationRate: number): string => {
  const { heatmap } = chartTheme.colors
  if (cooperationRate < 0.33) return heatmap.defect
  if (cooperationRate < 0.67) return heatmap.neutral
  return heatmap.cooperate
}