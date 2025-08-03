import React, { useRef, useEffect, useState, ReactNode } from 'react'
import { Skeleton } from '../loading/Skeleton'

interface ChartWrapperProps {
  width?: number
  height?: number
  className?: string
  loading?: boolean
  error?: Error | null
  children: (dimensions: { width: number; height: number }) => ReactNode
  minHeight?: number
  aspectRatio?: number
}

export const ChartWrapper: React.FC<ChartWrapperProps> = ({
  width: fixedWidth,
  height: fixedHeight,
  className = '',
  loading = false,
  error = null,
  children,
  minHeight = 300,
  aspectRatio = 16 / 9,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { width } = containerRef.current.getBoundingClientRect()
        const calculatedHeight = fixedHeight || Math.max(minHeight, width / aspectRatio)
        setDimensions({
          width: fixedWidth || width,
          height: calculatedHeight,
        })
      }
    }

    updateDimensions()
    const resizeObserver = new ResizeObserver(updateDimensions)
    
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current)
    }

    return () => {
      resizeObserver.disconnect()
    }
  }, [fixedWidth, fixedHeight, minHeight, aspectRatio])

  if (loading) {
    return (
      <div className={`w-full ${className}`}>
        <Skeleton className="w-full" style={{ height: minHeight }} />
      </div>
    )
  }

  if (error) {
    return (
      <div 
        className={`w-full flex items-center justify-center bg-red-50 border border-red-200 rounded-lg ${className}`}
        style={{ height: minHeight }}
      >
        <div className="text-center p-4">
          <p className="text-red-600 font-semibold">Error loading chart</p>
          <p className="text-red-500 text-sm mt-1">{error.message}</p>
        </div>
      </div>
    )
  }

  return (
    <div 
      ref={containerRef} 
      className={`w-full ${className}`}
      style={{ height: dimensions.height || minHeight }}
    >
      {dimensions.width > 0 && dimensions.height > 0 && children(dimensions)}
    </div>
  )
}