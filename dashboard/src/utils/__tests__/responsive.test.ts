import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  getViewportWidth,
  getViewportHeight,
  isBreakpoint,
  getChartDimensions,
  getChartMargins,
  debounce,
  isTouchDevice,
  getResponsiveFontSize,
  getLegendColumns,
  shouldUseDrawer
} from '../responsive';

describe('Responsive Utilities', () => {
  const originalInnerWidth = window.innerWidth;
  const originalInnerHeight = window.innerHeight;

  beforeEach(() => {
    // Reset window dimensions
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1024
    });
    Object.defineProperty(window, 'innerHeight', {
      writable: true,
      configurable: true,
      value: 768
    });
  });

  afterEach(() => {
    window.innerWidth = originalInnerWidth;
    window.innerHeight = originalInnerHeight;
  });

  describe('getViewportWidth', () => {
    it('returns current viewport width', () => {
      window.innerWidth = 1200;
      expect(getViewportWidth()).toBe(1200);
    });
  });

  describe('getViewportHeight', () => {
    it('returns current viewport height', () => {
      window.innerHeight = 800;
      expect(getViewportHeight()).toBe(800);
    });
  });

  describe('isBreakpoint', () => {
    it('correctly identifies breakpoints', () => {
      window.innerWidth = 320;
      expect(isBreakpoint('sm')).toBe(false);
      
      window.innerWidth = 640;
      expect(isBreakpoint('sm')).toBe(true);
      expect(isBreakpoint('md')).toBe(false);
      
      window.innerWidth = 768;
      expect(isBreakpoint('md')).toBe(true);
      expect(isBreakpoint('lg')).toBe(false);
    });
  });

  describe('getChartDimensions', () => {
    it('returns mobile dimensions for small viewports', () => {
      window.innerWidth = 320;
      const dims = getChartDimensions();
      expect(dims.width).toBeLessThanOrEqual(320);
      expect(dims.height).toBe(240);
    });

    it('returns tablet dimensions for medium viewports', () => {
      window.innerWidth = 768;
      const dims = getChartDimensions();
      expect(dims.width).toBeLessThanOrEqual(600);
      expect(dims.height).toBe(400);
    });

    it('returns desktop dimensions for large viewports', () => {
      window.innerWidth = 1200;
      const dims = getChartDimensions();
      expect(dims.width).toBeLessThanOrEqual(800);
      expect(dims.height).toBe(500);
    });

    it('respects container width when provided', () => {
      const dims = getChartDimensions(500);
      expect(dims.width).toBeLessThanOrEqual(500);
    });
  });

  describe('getChartMargins', () => {
    it('returns mobile margins for small viewports', () => {
      window.innerWidth = 320;
      const margins = getChartMargins();
      expect(margins.top).toBe(20);
      expect(margins.left).toBe(40);
    });

    it('returns tablet margins for medium viewports', () => {
      window.innerWidth = 768;
      const margins = getChartMargins();
      expect(margins.top).toBe(30);
      expect(margins.left).toBe(50);
    });

    it('returns desktop margins for large viewports', () => {
      window.innerWidth = 1200;
      const margins = getChartMargins();
      expect(margins.top).toBe(40);
      expect(margins.left).toBe(60);
    });
  });

  describe('debounce', () => {
    it('delays function execution', async () => {
      vi.useFakeTimers();
      const mockFn = vi.fn();
      const debouncedFn = debounce(mockFn, 100);

      debouncedFn();
      expect(mockFn).not.toHaveBeenCalled();

      vi.advanceTimersByTime(50);
      expect(mockFn).not.toHaveBeenCalled();

      vi.advanceTimersByTime(50);
      expect(mockFn).toHaveBeenCalledTimes(1);

      vi.useRealTimers();
    });

    it('cancels previous calls', async () => {
      vi.useFakeTimers();
      const mockFn = vi.fn();
      const debouncedFn = debounce(mockFn, 100);

      debouncedFn('first');
      vi.advanceTimersByTime(50);
      debouncedFn('second');
      vi.advanceTimersByTime(100);

      expect(mockFn).toHaveBeenCalledTimes(1);
      expect(mockFn).toHaveBeenCalledWith('second');

      vi.useRealTimers();
    });
  });

  describe('isTouchDevice', () => {
    it('detects touch support', () => {
      // Mock touch support
      Object.defineProperty(window, 'ontouchstart', {
        configurable: true,
        value: () => {}
      });
      expect(isTouchDevice()).toBe(true);

      // Remove touch support
      delete (window as any).ontouchstart;
      expect(isTouchDevice()).toBe(false);
    });
  });

  describe('getResponsiveFontSize', () => {
    it('scales font size for mobile', () => {
      window.innerWidth = 320;
      expect(getResponsiveFontSize(16)).toBe(16 * 0.85);
    });

    it('scales font size for small tablets', () => {
      window.innerWidth = 640;
      expect(getResponsiveFontSize(16)).toBe(16 * 0.9);
    });

    it('returns base size for desktop', () => {
      window.innerWidth = 1200;
      expect(getResponsiveFontSize(16)).toBe(16);
    });
  });

  describe('getLegendColumns', () => {
    it('returns 1 column for mobile', () => {
      window.innerWidth = 320;
      expect(getLegendColumns()).toBe(1);
    });

    it('returns 2 columns for small tablets', () => {
      window.innerWidth = 640;
      expect(getLegendColumns()).toBe(2);
    });

    it('returns 3 columns for large tablets', () => {
      window.innerWidth = 768;
      expect(getLegendColumns()).toBe(3);
    });

    it('returns 4 columns for desktop', () => {
      window.innerWidth = 1200;
      expect(getLegendColumns()).toBe(4);
    });
  });

  describe('shouldUseDrawer', () => {
    it('returns true for mobile viewports', () => {
      window.innerWidth = 500;
      expect(shouldUseDrawer()).toBe(true);
    });

    it('returns false for tablet and desktop viewports', () => {
      window.innerWidth = 768;
      expect(shouldUseDrawer()).toBe(false);
    });
  });
});