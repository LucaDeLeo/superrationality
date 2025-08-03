import React, { useEffect, useRef } from 'react'
import { chartTheme } from './chartTheme'

interface TooltipData {
  x: number
  y: number
  content: React.ReactNode
}

interface ChartTooltipProps {
  data: TooltipData | null
  containerRef?: React.RefObject<HTMLElement>
}

export const ChartTooltip: React.FC<ChartTooltipProps> = ({ data, containerRef }) => {
  const tooltipRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!data || !tooltipRef.current) return

    const tooltip = tooltipRef.current
    const { x, y } = data

    // Get container bounds for boundary detection
    const container = containerRef?.current || document.body
    const containerRect = container.getBoundingClientRect()
    const tooltipRect = tooltip.getBoundingClientRect()

    // Calculate position with boundary detection
    let left = x
    let top = y

    // Adjust horizontal position if tooltip would overflow
    if (x + tooltipRect.width > containerRect.right) {
      left = x - tooltipRect.width - 10
    }

    // Adjust vertical position if tooltip would overflow
    if (y + tooltipRect.height > containerRect.bottom) {
      top = y - tooltipRect.height - 10
    }

    // Ensure tooltip stays within container bounds
    left = Math.max(containerRect.left, Math.min(left, containerRect.right - tooltipRect.width))
    top = Math.max(containerRect.top, Math.min(top, containerRect.bottom - tooltipRect.height))

    tooltip.style.left = `${left}px`
    tooltip.style.top = `${top}px`
  }, [data, containerRef])

  if (!data) return null

  return (
    <div
      ref={tooltipRef}
      className="fixed z-50 pointer-events-none transition-opacity duration-200"
      style={{
        backgroundColor: chartTheme.tooltip.background,
        color: chartTheme.tooltip.color,
        padding: `${chartTheme.tooltip.padding}px`,
        borderRadius: `${chartTheme.tooltip.borderRadius}px`,
        fontSize: `${chartTheme.tooltip.fontSize}px`,
        boxShadow: chartTheme.tooltip.boxShadow,
      }}
    >
      {data.content}
    </div>
  )
}

// Hook for managing tooltip state
export const useChartTooltip = (containerRef?: React.RefObject<HTMLElement>) => {
  const [tooltipData, setTooltipData] = React.useState<TooltipData | null>(null)

  const showTooltip = (x: number, y: number, content: React.ReactNode) => {
    setTooltipData({ x, y, content })
  }

  const hideTooltip = () => {
    setTooltipData(null)
  }

  const tooltip = <ChartTooltip data={tooltipData} containerRef={containerRef} />

  return { showTooltip, hideTooltip, tooltip }
}