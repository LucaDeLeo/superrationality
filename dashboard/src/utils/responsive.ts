// Responsive breakpoints matching Tailwind CSS
export const breakpoints = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
} as const;

// Get current viewport width
export const getViewportWidth = (): number => {
  return window.innerWidth || document.documentElement.clientWidth;
};

// Get current viewport height
export const getViewportHeight = (): number => {
  return window.innerHeight || document.documentElement.clientHeight;
};

// Check if viewport matches breakpoint
export const isBreakpoint = (breakpoint: keyof typeof breakpoints): boolean => {
  return getViewportWidth() >= breakpoints[breakpoint];
};

// Get chart dimensions based on viewport
export const getChartDimensions = (containerWidth?: number): { width: number; height: number } => {
  const viewportWidth = containerWidth || getViewportWidth();
  
  // Mobile (< 640px)
  if (viewportWidth < breakpoints.sm) {
    return {
      width: Math.min(viewportWidth - 32, 320),
      height: 240
    };
  }
  
  // Tablet (640px - 1024px)
  if (viewportWidth < breakpoints.lg) {
    return {
      width: Math.min(viewportWidth - 48, 600),
      height: 400
    };
  }
  
  // Desktop (>= 1024px)
  return {
    width: Math.min(viewportWidth - 64, 800),
    height: 500
  };
};

// Get margin based on viewport
export const getChartMargins = () => {
  const viewportWidth = getViewportWidth();
  
  // Mobile
  if (viewportWidth < breakpoints.sm) {
    return { top: 20, right: 20, bottom: 40, left: 40 };
  }
  
  // Tablet
  if (viewportWidth < breakpoints.lg) {
    return { top: 30, right: 30, bottom: 50, left: 50 };
  }
  
  // Desktop
  return { top: 40, right: 40, bottom: 60, left: 60 };
};

// Debounce function for resize events
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};

// Hook-like function to handle resize
export const onResize = (callback: () => void): (() => void) => {
  const debouncedCallback = debounce(callback, 250);
  window.addEventListener('resize', debouncedCallback);
  
  // Return cleanup function
  return () => {
    window.removeEventListener('resize', debouncedCallback);
  };
};

// Touch event helpers
export const isTouchDevice = (): boolean => {
  return 'ontouchstart' in window || navigator.maxTouchPoints > 0;
};

// Scale font size based on viewport
export const getResponsiveFontSize = (baseSize: number): number => {
  const viewportWidth = getViewportWidth();
  
  if (viewportWidth < breakpoints.sm) {
    return baseSize * 0.85;
  }
  
  if (viewportWidth < breakpoints.md) {
    return baseSize * 0.9;
  }
  
  return baseSize;
};

// Get number of legend columns based on viewport
export const getLegendColumns = (): number => {
  const viewportWidth = getViewportWidth();
  
  if (viewportWidth < breakpoints.sm) {
    return 1;
  }
  
  if (viewportWidth < breakpoints.md) {
    return 2;
  }
  
  if (viewportWidth < breakpoints.lg) {
    return 3;
  }
  
  return 4;
};

// Mobile-friendly drawer state
export const shouldUseDrawer = (): boolean => {
  return getViewportWidth() < breakpoints.md;
};