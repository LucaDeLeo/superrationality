import * as d3 from 'd3'
import { chartTheme } from '@/components/visualizations/chartTheme'

// Format numbers with appropriate precision
export const formatNumber = (value: number, precision = 2): string => {
  if (Math.abs(value) >= 1000) {
    return d3.format('.2s')(value)
  }
  return value.toFixed(precision)
}

// Format percentage
export const formatPercent = (value: number): string => {
  return `${(value * 100).toFixed(1)}%`
}

// Create responsive SVG
export const createResponsiveSvg = (
  container: HTMLElement,
  width: number,
  height: number
) => {
  return d3
    .select(container)
    .append('svg')
    .attr('width', width)
    .attr('height', height)
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
}

// Add grid lines to a chart
export const addGridLines = (
  g: d3.Selection<any, any, any, any>,
  xScale: d3.ScaleLinear<number, number> | d3.ScaleTime<number, number>,
  yScale: d3.ScaleLinear<number, number>,
  width: number,
  height: number
) => {
  // Add X grid lines
  g.append('g')
    .attr('class', 'grid grid-x')
    .attr('transform', `translate(0,${height})`)
    .call(
      d3.axisBottom(xScale as any)
        .tickSize(-height)
        .tickFormat(() => '')
    )
    .style('stroke', chartTheme.gridLines.stroke)
    .style('stroke-dasharray', chartTheme.gridLines.strokeDasharray)
    .style('stroke-width', chartTheme.gridLines.strokeWidth)

  // Add Y grid lines
  g.append('g')
    .attr('class', 'grid grid-y')
    .call(
      d3.axisLeft(yScale)
        .tickSize(-width)
        .tickFormat(() => '')
    )
    .style('stroke', chartTheme.gridLines.stroke)
    .style('stroke-dasharray', chartTheme.gridLines.strokeDasharray)
    .style('stroke-width', chartTheme.gridLines.strokeWidth)
}

// Add chart title
export const addChartTitle = (
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  title: string,
  width: number,
  marginTop: number
) => {
  svg
    .append('text')
    .attr('class', 'chart-title')
    .attr('x', width / 2)
    .attr('y', marginTop / 2)
    .attr('text-anchor', 'middle')
    .style('font-size', `${chartTheme.fonts.sizes.title}px`)
    .style('font-weight', 'bold')
    .style('fill', chartTheme.colors.text)
    .text(title)
}

// Create a legend
export const createLegend = (
  container: d3.Selection<any, any, any, any>,
  items: { label: string; color: string }[],
  x: number,
  y: number
) => {
  const legend = container
    .append('g')
    .attr('class', 'legend')
    .attr('transform', `translate(${x},${y})`)

  const legendItems = legend
    .selectAll('.legend-item')
    .data(items)
    .enter()
    .append('g')
    .attr('class', 'legend-item')
    .attr('transform', (d, i) => `translate(0, ${i * 20})`)

  legendItems
    .append('rect')
    .attr('x', 0)
    .attr('y', -10)
    .attr('width', 15)
    .attr('height', 15)
    .style('fill', d => d.color)

  legendItems
    .append('text')
    .attr('x', 20)
    .attr('y', 0)
    .style('font-size', `${chartTheme.fonts.sizes.label}px`)
    .style('fill', chartTheme.colors.text)
    .text(d => d.label)

  return legend
}

// Transition configuration
export const getTransition = () => {
  return d3.transition()
    .duration(chartTheme.animation.duration)
    .ease(d3[chartTheme.animation.easing as keyof typeof d3] as any)
}

// Wrap text for long labels
export const wrapText = (
  text: d3.Selection<any, any, any, any>,
  width: number
) => {
  text.each(function() {
    const text = d3.select(this)
    const words = text.text().split(/\s+/).reverse()
    let word
    let line: string[] = []
    let lineNumber = 0
    const lineHeight = 1.1 // ems
    const y = text.attr('y')
    const dy = parseFloat(text.attr('dy') || '0')
    let tspan = text
      .text(null)
      .append('tspan')
      .attr('x', 0)
      .attr('y', y)
      .attr('dy', dy + 'em')

    while ((word = words.pop())) {
      line.push(word)
      tspan.text(line.join(' '))
      if ((tspan.node() as any)?.getComputedTextLength() > width) {
        line.pop()
        tspan.text(line.join(' '))
        line = [word]
        tspan = text
          .append('tspan')
          .attr('x', 0)
          .attr('y', y)
          .attr('dy', ++lineNumber * lineHeight + dy + 'em')
          .text(word)
      }
    }
  })
}

// Export chart as image
export const exportChart = async (
  svgElement: SVGSVGElement,
  filename: string,
  format: 'png' | 'svg' = 'png',
  scale = 2
) => {
  if (format === 'svg') {
    // Export as SVG
    const svgData = new XMLSerializer().serializeToString(svgElement)
    const blob = new Blob([svgData], { type: 'image/svg+xml' })
    const url = URL.createObjectURL(blob)
    
    const link = document.createElement('a')
    link.href = url
    link.download = `${filename}.svg`
    link.click()
    
    URL.revokeObjectURL(url)
  } else {
    // Export as PNG
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const svgData = new XMLSerializer().serializeToString(svgElement)
    const img = new Image()
    
    const svgBlob = new Blob([svgData], { type: 'image/svg+xml' })
    const url = URL.createObjectURL(svgBlob)
    
    img.onload = () => {
      canvas.width = img.width * scale
      canvas.height = img.height * scale
      ctx.scale(scale, scale)
      ctx.drawImage(img, 0, 0)
      
      canvas.toBlob((blob) => {
        if (blob) {
          const url = URL.createObjectURL(blob)
          const link = document.createElement('a')
          link.href = url
          link.download = `${filename}.png`
          link.click()
          URL.revokeObjectURL(url)
        }
      })
      
      URL.revokeObjectURL(url)
    }
    
    img.src = url
  }
}