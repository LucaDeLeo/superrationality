import React, { useEffect, useRef, useMemo } from 'react'
import * as d3 from 'd3'
import { ChartWrapper } from './ChartWrapper'
import { useChartTooltip } from './ChartTooltip'
import { chartTheme, getAgentColor } from './chartTheme'
import { 
  formatPercent, 
  addGridLines, 
  addChartTitle,
  createLegend,
  getTransition
} from '@/utils/chartHelpers'
import { 
  RoundSummary, 
  extractAgentCooperationRates,
  AgentCooperationData 
} from '@/utils/dataTransformers'

interface CooperationChartProps {
  data: RoundSummary[]
  title?: string
  loading?: boolean
  error?: Error | null
  selectedAgents?: string[]
  onAgentClick?: (agentId: string) => void
}

export const CooperationChart: React.FC<CooperationChartProps> = ({
  data,
  title = 'Cooperation Rates Over Rounds',
  loading = false,
  error = null,
  selectedAgents,
  onAgentClick
}) => {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const { showTooltip, hideTooltip, tooltip } = useChartTooltip(containerRef)

  // Transform data
  const cooperationData = useMemo(() => {
    if (!data || data.length === 0) return []
    const allData = extractAgentCooperationRates(data)
    
    // Filter by selected agents if provided
    if (selectedAgents && selectedAgents.length > 0) {
      return allData.filter(agent => selectedAgents.includes(agent.agentId))
    }
    
    return allData
  }, [data, selectedAgents])

  useEffect(() => {
    if (!svgRef.current || cooperationData.length === 0) return

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

    // Scales
    const xScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.round) as [number, number])
      .range([0, innerWidth])

    const yScale = d3.scaleLinear()
      .domain([0, 1])
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
      .call(d3.axisLeft(yScale).tickFormat(d => formatPercent(d as number)))
      .style('font-size', `${chartTheme.fonts.sizes.axis}px`)

    // Add axis labels
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left + 20)
      .attr('x', 0 - innerHeight / 2)
      .attr('text-anchor', 'middle')
      .style('font-size', `${chartTheme.fonts.sizes.axis}px`)
      .style('fill', chartTheme.colors.text)
      .text('Cooperation Rate')

    // Line generator
    const line = d3.line<{ round: number; cooperationRate: number }>()
      .x(d => xScale(d.round))
      .y(d => yScale(d.cooperationRate))
      .curve(d3.curveMonotoneX)

    // Draw lines for each agent
    const agentGroups = g.selectAll('.agent-line-group')
      .data(cooperationData)
      .enter()
      .append('g')
      .attr('class', 'agent-line-group')

    // Add lines with animation
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
        // Highlight line
        d3.select(this).attr('stroke-width', 4)
        
        // Show agent ID in tooltip
        const [x, y] = d3.pointer(event, svg.node())
        showTooltip(x, y, <div>{d.agentId}</div>)
      })
      .on('mouseout', function() {
        d3.select(this).attr('stroke-width', 2)
        hideTooltip()
      })
      .attr('stroke-dasharray', function() {
        return this.getTotalLength()
      })
      .attr('stroke-dashoffset', function() {
        return this.getTotalLength()
      })
      .transition(getTransition())
      .attr('stroke-dashoffset', 0)

    // Add dots for data points
    agentGroups.each(function(agentData, agentIndex) {
      const group = d3.select(this)
      
      group.selectAll('.data-point')
        .data(agentData.rounds)
        .enter()
        .append('circle')
        .attr('class', 'data-point')
        .attr('cx', d => xScale(d.round))
        .attr('cy', d => yScale(d.cooperationRate))
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
              <div>Cooperation: {formatPercent(d.cooperationRate)}</div>
              <div>Games: {d.totalGames}</div>
            </div>
          ))
        })
        .on('mouseout', function() {
          d3.select(this).attr('r', 4)
          hideTooltip()
        })
        .transition(getTransition())
        .delay((d, i) => i * 50)
        .attr('r', 4)
    })

    // Add overall cooperation rate line
    const overallLine = data.map(d => ({
      round: d.round,
      cooperationRate: d.cooperation_rate
    }))

    g.append('path')
      .datum(overallLine)
      .attr('class', 'overall-line')
      .attr('fill', 'none')
      .attr('stroke', chartTheme.colors.neutral)
      .attr('stroke-width', 3)
      .attr('stroke-dasharray', '5,5')
      .attr('d', line)

    // Add legend
    const legendItems = [
      { label: 'Overall', color: chartTheme.colors.neutral },
      ...cooperationData.slice(0, 10).map((agent, i) => ({
        label: agent.agentId,
        color: getAgentColor(i)
      }))
    ]

    createLegend(g, legendItems, innerWidth - 100, 0)

    // Add title
    addChartTitle(svg, title, width, margin.top)

  }, [cooperationData, data, title, showTooltip, hideTooltip, onAgentClick])

  return (
    <div ref={containerRef} className="relative">
      <ChartWrapper loading={loading} error={error} minHeight={400}>
        {({ width, height }) => (
          <svg 
            ref={svgRef} 
            width={width} 
            height={height}
            className="cooperation-chart"
          />
        )}
      </ChartWrapper>
      {tooltip}
    </div>
  )
}