import React, { useEffect, useRef, useMemo } from 'react'
import * as d3 from 'd3'
import { ChartWrapper } from './ChartWrapper'
import { useChartTooltip } from './ChartTooltip'
import { chartTheme, getAgentColor } from './chartTheme'
import { 
  formatNumber, 
  addGridLines, 
  addChartTitle,
  createLegend,
  getTransition
} from '@/utils/chartHelpers'
import { 
  RoundSummary, 
  calculateAgentScores,
  AgentScoreData 
} from '@/utils/dataTransformers'

interface ScoreProgressionChartProps {
  data: RoundSummary[]
  title?: string
  loading?: boolean
  error?: Error | null
  chartType?: 'line' | 'area' | 'stacked'
  selectedAgents?: string[]
  showAnimation?: boolean
  onAgentClick?: (agentId: string) => void
}

export const ScoreProgressionChart: React.FC<ScoreProgressionChartProps> = ({
  data,
  title = 'Score Progression Across Rounds',
  loading = false,
  error = null,
  chartType = 'area',
  selectedAgents,
  showAnimation = true,
  onAgentClick
}) => {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const { showTooltip, hideTooltip, tooltip } = useChartTooltip(containerRef)

  // Transform data
  const scoreData = useMemo(() => {
    if (!data || data.length === 0) return []
    const allData = calculateAgentScores(data)
    
    // Filter by selected agents if provided
    if (selectedAgents && selectedAgents.length > 0) {
      return allData.filter(agent => selectedAgents.includes(agent.agentId))
    }
    
    return allData
  }, [data, selectedAgents])

  useEffect(() => {
    if (!svgRef.current || scoreData.length === 0) return

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

    // Get all rounds and max score
    const allRounds = Array.from(new Set(data.map(d => d.round))).sort()
    const maxScore = d3.max(scoreData, agent => 
      d3.max(agent.rounds, r => r.cumulativeScore)
    ) || 0

    // Scales
    const xScale = d3.scaleLinear()
      .domain([Math.min(...allRounds), Math.max(...allRounds)])
      .range([0, innerWidth])

    const yScale = d3.scaleLinear()
      .domain([0, maxScore * 1.1]) // Add 10% padding
      .range([innerHeight, 0])

    // Add grid lines
    addGridLines(g, xScale, yScale, innerWidth, innerHeight)

    // Add axes
    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).tickFormat(d => `Round ${d}`))
      .style('font-size', `${chartTheme.fonts.sizes.axis}px`)

    g.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(yScale).tickFormat(d => formatNumber(d as number)))
      .style('font-size', `${chartTheme.fonts.sizes.axis}px`)

    // Add axis labels
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left + 20)
      .attr('x', 0 - innerHeight / 2)
      .attr('text-anchor', 'middle')
      .style('font-size', `${chartTheme.fonts.sizes.axis}px`)
      .style('fill', chartTheme.colors.text)
      .text('Cumulative Score')

    if (chartType === 'stacked') {
      // Prepare stacked data
      const stackData = allRounds.map(round => {
        const roundData: any = { round }
        scoreData.forEach(agent => {
          const roundScore = agent.rounds.find(r => r.round === round)
          roundData[agent.agentId] = roundScore ? roundScore.score : 0
        })
        return roundData
      })

      const stack = d3.stack()
        .keys(scoreData.map(d => d.agentId))
        .order(d3.stackOrderNone)
        .offset(d3.stackOffsetNone)

      const series = stack(stackData)

      // Area generator for stacked
      const area = d3.area<any>()
        .x(d => xScale(d.data.round))
        .y0(d => yScale(d[0]))
        .y1(d => yScale(d[1]))
        .curve(d3.curveMonotoneX)

      // Draw stacked areas
      g.selectAll('.stack-area')
        .data(series)
        .enter()
        .append('path')
        .attr('class', 'stack-area')
        .attr('fill', (d, i) => getAgentColor(i))
        .attr('opacity', 0.8)
        .attr('d', area)
        .style('cursor', onAgentClick ? 'pointer' : 'default')
        .on('click', function(event, d) {
          if (onAgentClick) {
            onAgentClick(d.key)
          }
        })
        .on('mouseover', function(event, d) {
          d3.select(this).attr('opacity', 1)
          
          const [x, y] = d3.pointer(event, svg.node())
          showTooltip(x, y, <div>{d.key}</div>)
        })
        .on('mouseout', function() {
          d3.select(this).attr('opacity', 0.8)
          hideTooltip()
        })

    } else {
      // Line or area chart
      const line = d3.line<{ round: number; cumulativeScore: number }>()
        .x(d => xScale(d.round))
        .y(d => yScale(d.cumulativeScore))
        .curve(d3.curveMonotoneX)

      const area = d3.area<{ round: number; cumulativeScore: number }>()
        .x(d => xScale(d.round))
        .y0(innerHeight)
        .y1(d => yScale(d.cumulativeScore))
        .curve(d3.curveMonotoneX)

      // Draw areas or lines for each agent
      const agentGroups = g.selectAll('.agent-score-group')
        .data(scoreData)
        .enter()
        .append('g')
        .attr('class', 'agent-score-group')

      if (chartType === 'area') {
        // Draw areas
        agentGroups
          .append('path')
          .attr('class', 'agent-area')
          .attr('fill', (d, i) => getAgentColor(i))
          .attr('opacity', 0.3)
          .attr('d', d => area(d.rounds))
      }

      // Draw lines
      agentGroups
        .append('path')
        .attr('class', 'agent-line')
        .attr('fill', 'none')
        .attr('stroke', (d, i) => getAgentColor(i))
        .attr('stroke-width', 2)
        .attr('d', d => line(d.rounds))
        .style('cursor', onAgentClick ? 'pointer' : 'default')
        .on('click', function(event, d) {
          if (onAgentClick) {
            onAgentClick(d.agentId)
          }
        })
        .on('mouseover', function(event, d) {
          d3.select(this).attr('stroke-width', 4)
          
          const [x, y] = d3.pointer(event, svg.node())
          showTooltip(x, y, <div>{d.agentId}</div>)
        })
        .on('mouseout', function() {
          d3.select(this).attr('stroke-width', 2)
          hideTooltip()
        })

      // Add animation if enabled
      if (showAnimation) {
        agentGroups.selectAll('path')
          .attr('stroke-dasharray', function() {
            return this.getTotalLength()
          })
          .attr('stroke-dashoffset', function() {
            return this.getTotalLength()
          })
          .transition(getTransition())
          .attr('stroke-dashoffset', 0)
      }

      // Add dots for data points
      agentGroups.each(function(agentData, agentIndex) {
        const group = d3.select(this)
        
        group.selectAll('.data-point')
          .data(agentData.rounds)
          .enter()
          .append('circle')
          .attr('class', 'data-point')
          .attr('cx', d => xScale(d.round))
          .attr('cy', d => yScale(d.cumulativeScore))
          .attr('r', 0)
          .attr('fill', getAgentColor(agentIndex))
          .style('cursor', 'pointer')
          .on('mouseover', function(event, d) {
            d3.select(this).attr('r', 6)
            
            const [x, y] = d3.pointer(event, svg.node())
            showTooltip(x, y, (
              <div>
                <div><strong>{agentData.agentId}</strong></div>
                <div>Round {d.round}</div>
                <div>Round Score: {formatNumber(d.score)}</div>
                <div>Total Score: {formatNumber(d.cumulativeScore)}</div>
              </div>
            ))
          })
          .on('mouseout', function() {
            d3.select(this).attr('r', 4)
            hideTooltip()
          })
          .transition(getTransition())
          .delay((d, i) => showAnimation ? i * 50 : 0)
          .attr('r', 4)
      })
    }

    // Add range selector (simplified version)
    const brush = d3.brushX()
      .extent([[0, 0], [innerWidth, 30]])
      .on('end', function(event) {
        if (!event.selection) return
        
        const [x0, x1] = event.selection as [number, number]
        const newDomain = [xScale.invert(x0), xScale.invert(x1)]
        
        // Update x scale domain
        xScale.domain(newDomain)
        
        // Update visualization
        // In a real implementation, this would trigger a re-render
        console.log('New range:', newDomain)
      })

    const brushGroup = g.append('g')
      .attr('class', 'brush')
      .attr('transform', `translate(0,${innerHeight + 40})`)
      .call(brush)

    // Style the brush
    brushGroup.selectAll('.selection')
      .style('fill', chartTheme.colors.primary)
      .style('fill-opacity', 0.2)

    // Add legend
    const legendItems = scoreData.slice(0, 10).map((agent, i) => ({
      label: agent.agentId,
      color: getAgentColor(i)
    }))

    createLegend(g, legendItems, innerWidth - 100, 0)

    // Add title
    addChartTitle(svg, title, width, margin.top)

  }, [scoreData, data, chartType, title, showAnimation, showTooltip, hideTooltip, onAgentClick])

  return (
    <div ref={containerRef} className="relative">
      <ChartWrapper loading={loading} error={error} minHeight={450}>
        {({ width, height }) => (
          <svg 
            ref={svgRef} 
            width={width} 
            height={height}
            className="score-progression-chart"
          />
        )}
      </ChartWrapper>
      {tooltip}
    </div>
  )
}