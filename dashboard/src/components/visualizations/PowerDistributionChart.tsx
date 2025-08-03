import React, { useEffect, useRef, useMemo } from 'react'
import * as d3 from 'd3'
import { ChartWrapper } from './ChartWrapper'
import { useChartTooltip } from './ChartTooltip'
import { chartTheme } from './chartTheme'
import { 
  formatNumber, 
  addGridLines, 
  addChartTitle,
  getTransition
} from '@/utils/chartHelpers'
import { RoundSummary } from '@/utils/dataTransformers'

interface PowerDistributionChartProps {
  data: RoundSummary[]
  title?: string
  loading?: boolean
  error?: Error | null
  chartType?: 'boxplot' | 'violin' | 'histogram'
  selectedRound?: number
  showEvolution?: boolean
  onRoundChange?: (round: number) => void
}

interface BoxPlotData {
  round: number
  min: number
  q1: number
  median: number
  q3: number
  max: number
  mean: number
  outliers: number[]
}

export const PowerDistributionChart: React.FC<PowerDistributionChartProps> = ({
  data,
  title = 'Agent Power Distribution',
  loading = false,
  error = null,
  chartType = 'boxplot',
  selectedRound,
  showEvolution = true,
  onRoundChange
}) => {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const { showTooltip, hideTooltip, tooltip } = useChartTooltip(containerRef)

  // Process power distribution data
  const powerData = useMemo(() => {
    if (!data || data.length === 0) return []
    
    return data.map(round => {
      const dist = round.power_distribution
      
      // Calculate quartiles (simplified - in real implementation would use actual agent data)
      const q1 = dist.mean - dist.std
      const q3 = dist.mean + dist.std
      const median = dist.mean
      
      return {
        round: round.round,
        min: dist.min,
        q1,
        median,
        q3,
        max: dist.max,
        mean: dist.mean,
        outliers: [] // Would be calculated from actual agent data
      } as BoxPlotData
    })
  }, [data])

  const currentRound = selectedRound || (powerData.length > 0 ? powerData[powerData.length - 1].round : 1)

  useEffect(() => {
    if (!svgRef.current || powerData.length === 0) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = svgRef.current.clientWidth
    const height = svgRef.current.clientHeight
    const margin = chartTheme.margins

    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // Create main group
    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    if (showEvolution) {
      // Show all rounds
      const xScale = d3.scaleBand()
        .domain(powerData.map(d => String(d.round)))
        .range([0, innerWidth])
        .padding(0.2)

      const yScale = d3.scaleLinear()
        .domain([
          d3.min(powerData, d => d.min) || 0,
          d3.max(powerData, d => d.max) || 200
        ])
        .nice()
        .range([innerHeight, 0])

      // Add grid lines
      addGridLines(g, xScale as any, yScale, innerWidth, innerHeight)

      // Add axes
      g.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale).tickFormat(d => `Round ${d}`))
        .style('font-size', `${chartTheme.fonts.sizes.axis}px`)

      g.append('g')
        .attr('class', 'y-axis')
        .call(d3.axisLeft(yScale))
        .style('font-size', `${chartTheme.fonts.sizes.axis}px`)

      // Add axis label
      g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', 0 - margin.left + 20)
        .attr('x', 0 - innerHeight / 2)
        .attr('text-anchor', 'middle')
        .style('font-size', `${chartTheme.fonts.sizes.axis}px`)
        .style('fill', chartTheme.colors.text)
        .text('Agent Power')

      if (chartType === 'boxplot') {
        // Draw box plots
        const boxWidth = xScale.bandwidth() * 0.7

        const boxes = g.selectAll('.box')
          .data(powerData)
          .enter()
          .append('g')
          .attr('class', 'box')
          .attr('transform', d => `translate(${xScale(String(d.round))},0)`)

        // Vertical lines (min to max)
        boxes.append('line')
          .attr('x1', xScale.bandwidth() / 2)
          .attr('x2', xScale.bandwidth() / 2)
          .attr('y1', d => yScale(d.min))
          .attr('y2', d => yScale(d.max))
          .attr('stroke', chartTheme.colors.text)
          .attr('stroke-width', 1)

        // Box (q1 to q3)
        boxes.append('rect')
          .attr('x', (xScale.bandwidth() - boxWidth) / 2)
          .attr('y', d => yScale(d.q3))
          .attr('width', boxWidth)
          .attr('height', d => yScale(d.q1) - yScale(d.q3))
          .attr('fill', chartTheme.colors.primary)
          .attr('fill-opacity', 0.6)
          .attr('stroke', chartTheme.colors.primary)
          .attr('stroke-width', 2)
          .style('cursor', 'pointer')
          .on('click', function(event, d) {
            if (onRoundChange) {
              onRoundChange(d.round)
            }
          })
          .on('mouseover', function(event, d) {
            d3.select(this).attr('fill-opacity', 0.8)
            
            const [x, y] = d3.pointer(event, svg.node())
            showTooltip(x, y, (
              <div>
                <div><strong>Round {d.round}</strong></div>
                <div>Max: {formatNumber(d.max)}</div>
                <div>Q3: {formatNumber(d.q3)}</div>
                <div>Median: {formatNumber(d.median)}</div>
                <div>Q1: {formatNumber(d.q1)}</div>
                <div>Min: {formatNumber(d.min)}</div>
                <div>Mean: {formatNumber(d.mean)}</div>
              </div>
            ))
          })
          .on('mouseout', function() {
            d3.select(this).attr('fill-opacity', 0.6)
            hideTooltip()
          })

        // Median line
        boxes.append('line')
          .attr('x1', (xScale.bandwidth() - boxWidth) / 2)
          .attr('x2', (xScale.bandwidth() - boxWidth) / 2 + boxWidth)
          .attr('y1', d => yScale(d.median))
          .attr('y2', d => yScale(d.median))
          .attr('stroke', chartTheme.colors.text)
          .attr('stroke-width', 2)

        // Mean marker
        boxes.append('circle')
          .attr('cx', xScale.bandwidth() / 2)
          .attr('cy', d => yScale(d.mean))
          .attr('r', 4)
          .attr('fill', chartTheme.colors.danger)
          .attr('stroke', 'white')
          .attr('stroke-width', 1)

        // Whisker caps
        boxes.append('line')
          .attr('x1', xScale.bandwidth() / 2 - boxWidth / 4)
          .attr('x2', xScale.bandwidth() / 2 + boxWidth / 4)
          .attr('y1', d => yScale(d.min))
          .attr('y2', d => yScale(d.min))
          .attr('stroke', chartTheme.colors.text)
          .attr('stroke-width', 1)

        boxes.append('line')
          .attr('x1', xScale.bandwidth() / 2 - boxWidth / 4)
          .attr('x2', xScale.bandwidth() / 2 + boxWidth / 4)
          .attr('y1', d => yScale(d.max))
          .attr('y2', d => yScale(d.max))
          .attr('stroke', chartTheme.colors.text)
          .attr('stroke-width', 1)
      } else if (chartType === 'violin') {
        // Simplified violin plot (would need actual distribution data)
        const violinWidth = xScale.bandwidth() * 0.8

        powerData.forEach((roundData, i) => {
          const violinGroup = g.append('g')
            .attr('transform', `translate(${xScale(String(roundData.round))},0)`)

          // Create a normal distribution approximation
          const numPoints = 50
          const points: [number, number][] = []
          
          for (let i = 0; i <= numPoints; i++) {
            const t = i / numPoints
            const y = roundData.min + t * (roundData.max - roundData.min)
            
            // Simple gaussian-like shape
            const normalizedY = (y - roundData.mean) / (roundData.max - roundData.min)
            const width = Math.exp(-normalizedY * normalizedY * 4) * violinWidth / 2
            
            points.push([width, y])
          }

          // Create area generator
          const area = d3.area<[number, number]>()
            .x0(d => xScale.bandwidth() / 2 - d[0])
            .x1(d => xScale.bandwidth() / 2 + d[0])
            .y(d => yScale(d[1]))
            .curve(d3.curveBasis)

          violinGroup.append('path')
            .datum(points)
            .attr('d', area)
            .attr('fill', chartTheme.colors.primary)
            .attr('fill-opacity', 0.6)
            .attr('stroke', chartTheme.colors.primary)
            .attr('stroke-width', 2)

          // Add median line
          violinGroup.append('line')
            .attr('x1', xScale.bandwidth() / 2 - violinWidth / 4)
            .attr('x2', xScale.bandwidth() / 2 + violinWidth / 4)
            .attr('y1', yScale(roundData.median))
            .attr('y2', yScale(roundData.median))
            .attr('stroke', 'white')
            .attr('stroke-width', 2)
        })
      }

      // Add time slider
      const sliderHeight = 30
      const sliderY = innerHeight + 60

      const sliderScale = d3.scaleLinear()
        .domain([powerData[0].round, powerData[powerData.length - 1].round])
        .range([0, innerWidth])
        .clamp(true)

      const slider = g.append('g')
        .attr('class', 'slider')
        .attr('transform', `translate(0,${sliderY})`)

      slider.append('line')
        .attr('class', 'track')
        .attr('x1', sliderScale.range()[0])
        .attr('x2', sliderScale.range()[1])
        .style('stroke', '#e5e7eb')
        .style('stroke-width', 6)
        .style('stroke-linecap', 'round')

      slider.append('line')
        .attr('class', 'track-filled')
        .attr('x1', sliderScale.range()[0])
        .attr('x2', sliderScale(currentRound))
        .style('stroke', chartTheme.colors.primary)
        .style('stroke-width', 6)
        .style('stroke-linecap', 'round')

      const handle = slider.append('circle')
        .attr('class', 'handle')
        .attr('cx', sliderScale(currentRound))
        .attr('cy', 0)
        .attr('r', 8)
        .style('fill', chartTheme.colors.primary)
        .style('stroke', 'white')
        .style('stroke-width', 2)
        .style('cursor', 'pointer')
        .call(
          d3.drag<SVGCircleElement, unknown>()
            .on('drag', function(event) {
              const x = Math.max(0, Math.min(innerWidth, event.x))
              const round = Math.round(sliderScale.invert(x))
              
              d3.select(this).attr('cx', sliderScale(round))
              slider.select('.track-filled').attr('x2', sliderScale(round))
              
              if (onRoundChange) {
                onRoundChange(round)
              }
            })
        )

      // Add slider label
      slider.append('text')
        .attr('x', innerWidth / 2)
        .attr('y', 25)
        .attr('text-anchor', 'middle')
        .style('font-size', `${chartTheme.fonts.sizes.label}px`)
        .style('fill', chartTheme.colors.text)
        .text('Drag to view power distribution evolution')

    } else {
      // Show single round (histogram)
      const currentData = powerData.find(d => d.round === currentRound)
      if (!currentData) return

      // Create histogram bins (simplified)
      const numBins = 20
      const binWidth = (currentData.max - currentData.min) / numBins
      const bins = Array.from({ length: numBins }, (_, i) => {
        const x0 = currentData.min + i * binWidth
        const x1 = x0 + binWidth
        
        // Simulate frequency based on normal distribution
        const center = (x0 + x1) / 2
        const normalizedX = (center - currentData.mean) / (currentData.std || 1)
        const frequency = Math.exp(-normalizedX * normalizedX / 2) * 10
        
        return { x0, x1, frequency }
      })

      const xScale = d3.scaleLinear()
        .domain([currentData.min, currentData.max])
        .range([0, innerWidth])

      const yScale = d3.scaleLinear()
        .domain([0, d3.max(bins, d => d.frequency) || 10])
        .range([innerHeight, 0])

      // Add axes
      g.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale))
        .style('font-size', `${chartTheme.fonts.sizes.axis}px`)

      g.append('g')
        .attr('class', 'y-axis')
        .call(d3.axisLeft(yScale))
        .style('font-size', `${chartTheme.fonts.sizes.axis}px`)

      // Draw histogram bars
      g.selectAll('.bar')
        .data(bins)
        .enter()
        .append('rect')
        .attr('class', 'bar')
        .attr('x', d => xScale(d.x0))
        .attr('y', d => yScale(d.frequency))
        .attr('width', d => xScale(d.x1) - xScale(d.x0) - 1)
        .attr('height', d => innerHeight - yScale(d.frequency))
        .attr('fill', chartTheme.colors.primary)
        .attr('fill-opacity', 0.7)
        .on('mouseover', function(event, d) {
          d3.select(this).attr('fill-opacity', 0.9)
          
          const [x, y] = d3.pointer(event, svg.node())
          showTooltip(x, y, (
            <div>
              <div>Power Range: {formatNumber(d.x0)} - {formatNumber(d.x1)}</div>
              <div>Frequency: {formatNumber(d.frequency)}</div>
            </div>
          ))
        })
        .on('mouseout', function() {
          d3.select(this).attr('fill-opacity', 0.7)
          hideTooltip()
        })
    }

    // Add statistical summary
    const summaryData = powerData.find(d => d.round === currentRound)
    if (summaryData) {
      const summaryGroup = g.append('g')
        .attr('class', 'summary')
        .attr('transform', `translate(${innerWidth - 150}, 20)`)

      const summaryBox = summaryGroup.append('rect')
        .attr('x', -10)
        .attr('y', -10)
        .attr('width', 160)
        .attr('height', 100)
        .attr('fill', 'white')
        .attr('stroke', '#e5e7eb')
        .attr('stroke-width', 1)
        .attr('rx', 4)

      const summaryText = [
        `Round ${summaryData.round}`,
        `Mean: ${formatNumber(summaryData.mean)}`,
        `Median: ${formatNumber(summaryData.median)}`,
        `Min: ${formatNumber(summaryData.min)}`,
        `Max: ${formatNumber(summaryData.max)}`
      ]

      summaryText.forEach((text, i) => {
        summaryGroup.append('text')
          .attr('x', 0)
          .attr('y', i * 18 + 5)
          .style('font-size', `${chartTheme.fonts.sizes.label}px`)
          .style('font-weight', i === 0 ? 'bold' : 'normal')
          .text(text)
      })
    }

    // Add title
    addChartTitle(svg, title, width, margin.top)

  }, [powerData, chartType, currentRound, showEvolution, title, showTooltip, hideTooltip, onRoundChange])

  return (
    <div ref={containerRef} className="relative">
      <ChartWrapper loading={loading} error={error} minHeight={500}>
        {({ width, height }) => (
          <svg 
            ref={svgRef} 
            width={width} 
            height={height}
            className="power-distribution-chart"
          />
        )}
      </ChartWrapper>
      {tooltip}
    </div>
  )
}