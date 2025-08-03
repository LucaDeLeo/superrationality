import React, { useEffect, useRef, useMemo } from 'react'
import * as d3 from 'd3'
import { ChartWrapper } from './ChartWrapper'
import { useChartTooltip } from './ChartTooltip'
import { chartTheme } from './chartTheme'
import { formatNumber, addChartTitle } from '@/utils/chartHelpers'
import { 
  RoundSummary, 
  createMatchupMatrix,
  AgentMatchupData 
} from '@/utils/dataTransformers'

interface MatchupHeatMapProps {
  data: RoundSummary[]
  title?: string
  loading?: boolean
  error?: Error | null
  colorScale?: 'winRate' | 'cooperationRate' | 'scoreDiff'
  onCellClick?: (matchup: AgentMatchupData) => void
}

export const MatchupHeatMap: React.FC<MatchupHeatMapProps> = ({
  data,
  title = 'Agent vs Agent Matchup Results',
  loading = false,
  error = null,
  colorScale = 'winRate',
  onCellClick
}) => {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const { showTooltip, hideTooltip, tooltip } = useChartTooltip(containerRef)

  // Transform data
  const { matchupData, agents } = useMemo(() => {
    if (!data || data.length === 0) return { matchupData: [], agents: [] }
    
    const matrix = createMatchupMatrix(data)
    const agentSet = new Set<string>()
    
    matrix.forEach(matchup => {
      agentSet.add(matchup.agent1Id)
      agentSet.add(matchup.agent2Id)
    })
    
    return {
      matchupData: matrix,
      agents: Array.from(agentSet).sort()
    }
  }, [data])

  useEffect(() => {
    if (!svgRef.current || matchupData.length === 0) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = svgRef.current.clientWidth
    const height = svgRef.current.clientHeight
    const margin = { ...chartTheme.margins, left: 120, bottom: 120 }

    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // Calculate cell size
    const cellSize = Math.min(innerWidth / agents.length, innerHeight / agents.length)
    const actualWidth = cellSize * agents.length
    const actualHeight = cellSize * agents.length

    // Center the heat map
    const xOffset = (innerWidth - actualWidth) / 2
    const yOffset = (innerHeight - actualHeight) / 2

    // Create main group
    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left + xOffset},${margin.top + yOffset})`)

    // Create scales
    const xScale = d3.scaleBand()
      .domain(agents)
      .range([0, actualWidth])
      .padding(0.05)

    const yScale = d3.scaleBand()
      .domain(agents)
      .range([0, actualHeight])
      .padding(0.05)

    // Color scale based on selected metric
    let colorScaleFn: d3.ScaleSequential<string>
    let colorDomain: [number, number]
    
    switch (colorScale) {
      case 'cooperationRate':
        colorDomain = [0, 1]
        colorScaleFn = d3.scaleSequential()
          .domain(colorDomain)
          .interpolator(d3.interpolateRdYlGn)
        break
      case 'scoreDiff':
        const maxDiff = d3.max(matchupData, d => Math.abs(d.avgScoreDiff)) || 5
        colorDomain = [-maxDiff, maxDiff]
        colorScaleFn = d3.scaleSequential()
          .domain(colorDomain)
          .interpolator(d3.interpolateRdBu)
        break
      case 'winRate':
      default:
        colorDomain = [0, 1]
        colorScaleFn = d3.scaleSequential()
          .domain(colorDomain)
          .interpolator(d3.interpolateRdYlGn)
    }

    // Create a map for quick lookup
    const matchupMap = new Map<string, AgentMatchupData>()
    matchupData.forEach(matchup => {
      const key1 = `${matchup.agent1Id}_${matchup.agent2Id}`
      const key2 = `${matchup.agent2Id}_${matchup.agent1Id}`
      matchupMap.set(key1, matchup)
      matchupMap.set(key2, matchup)
    })

    // Create cells
    const cells = g.selectAll('.cell')
      .data(agents.flatMap(agent1 => 
        agents.map(agent2 => ({ agent1, agent2 }))
      ))
      .enter()
      .append('g')
      .attr('class', 'cell')
      .attr('transform', d => 
        `translate(${xScale(d.agent2)},${yScale(d.agent1)})`
      )

    // Add rectangles
    cells.append('rect')
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('fill', d => {
        if (d.agent1 === d.agent2) {
          return chartTheme.colors.neutral
        }
        
        const matchup = matchupMap.get(`${d.agent1}_${d.agent2}`)
        if (!matchup || matchup.totalGames === 0) {
          return '#f3f4f6' // gray-100
        }

        let value: number
        if (colorScale === 'cooperationRate') {
          value = matchup.cooperationRate
        } else if (colorScale === 'scoreDiff') {
          value = d.agent1 === matchup.agent1Id 
            ? matchup.avgScoreDiff 
            : -matchup.avgScoreDiff
        } else {
          // Win rate from perspective of row agent
          const winRate = d.agent1 === matchup.agent1Id
            ? matchup.wins / matchup.totalGames
            : matchup.losses / matchup.totalGames
          value = winRate
        }

        return colorScaleFn(value)
      })
      .attr('stroke', '#e5e7eb')
      .attr('stroke-width', 1)
      .style('cursor', d => d.agent1 !== d.agent2 && onCellClick ? 'pointer' : 'default')
      .on('click', function(event, d) {
        if (d.agent1 !== d.agent2 && onCellClick) {
          const matchup = matchupMap.get(`${d.agent1}_${d.agent2}`)
          if (matchup) {
            onCellClick(matchup)
          }
        }
      })
      .on('mouseover', function(event, d) {
        if (d.agent1 === d.agent2) return

        d3.select(this)
          .attr('stroke', chartTheme.colors.primary)
          .attr('stroke-width', 2)

        const matchup = matchupMap.get(`${d.agent1}_${d.agent2}`)
        if (!matchup || matchup.totalGames === 0) {
          const [x, y] = d3.pointer(event, svg.node())
          showTooltip(x, y, (
            <div>
              <div><strong>{d.agent1} vs {d.agent2}</strong></div>
              <div>No games played</div>
            </div>
          ))
          return
        }

        // Calculate values from perspective of row agent
        const isAgent1 = d.agent1 === matchup.agent1Id
        const wins = isAgent1 ? matchup.wins : matchup.losses
        const losses = isAgent1 ? matchup.losses : matchup.wins
        const winRate = matchup.totalGames > 0 ? wins / matchup.totalGames : 0
        const avgScoreDiff = isAgent1 ? matchup.avgScoreDiff : -matchup.avgScoreDiff

        const [x, y] = d3.pointer(event, svg.node())
        showTooltip(x, y, (
          <div>
            <div><strong>{d.agent1} vs {d.agent2}</strong></div>
            <div>Games: {matchup.totalGames}</div>
            <div>Win Rate: {(winRate * 100).toFixed(1)}%</div>
            <div>W-L-D: {wins}-{losses}-{matchup.draws}</div>
            <div>Avg Score Diff: {formatNumber(avgScoreDiff)}</div>
            <div>Mutual Cooperation: {(matchup.cooperationRate * 100).toFixed(1)}%</div>
          </div>
        ))
      })
      .on('mouseout', function() {
        d3.select(this)
          .attr('stroke', '#e5e7eb')
          .attr('stroke-width', 1)
        hideTooltip()
      })

    // Add text labels for diagonal
    cells.filter(d => d.agent1 === d.agent2)
      .append('text')
      .attr('x', xScale.bandwidth() / 2)
      .attr('y', yScale.bandwidth() / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .style('fill', 'white')
      .style('font-size', '10px')
      .style('font-weight', 'bold')
      .text('SELF')

    // Add x-axis labels
    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${actualHeight})`)
      .selectAll('text')
      .data(agents)
      .enter()
      .append('text')
      .attr('x', d => (xScale(d) || 0) + xScale.bandwidth() / 2)
      .attr('y', 10)
      .attr('text-anchor', 'start')
      .attr('transform', d => 
        `rotate(45,${(xScale(d) || 0) + xScale.bandwidth() / 2},10)`
      )
      .style('font-size', `${chartTheme.fonts.sizes.label}px`)
      .text(d => d)

    // Add y-axis labels
    g.append('g')
      .attr('class', 'y-axis')
      .selectAll('text')
      .data(agents)
      .enter()
      .append('text')
      .attr('x', -10)
      .attr('y', d => (yScale(d) || 0) + yScale.bandwidth() / 2)
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'middle')
      .style('font-size', `${chartTheme.fonts.sizes.label}px`)
      .text(d => d)

    // Add color legend
    const legendWidth = 200
    const legendHeight = 20
    const legendX = actualWidth + 40
    const legendY = actualHeight / 2 - legendHeight / 2

    const legendScale = d3.scaleLinear()
      .domain(colorDomain)
      .range([0, legendWidth])

    const legendAxis = d3.axisBottom(legendScale)
      .ticks(5)
      .tickFormat(d => {
        if (colorScale === 'cooperationRate' || colorScale === 'winRate') {
          return `${(d as number * 100).toFixed(0)}%`
        }
        return formatNumber(d as number)
      })

    // Create gradient
    const gradientId = `gradient-${colorScale}`
    const gradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', gradientId)
      .attr('x1', '0%')
      .attr('x2', '100%')
      .attr('y1', '0%')
      .attr('y2', '0%')

    // Add gradient stops
    const numStops = 10
    for (let i = 0; i <= numStops; i++) {
      const t = i / numStops
      const value = colorDomain[0] + t * (colorDomain[1] - colorDomain[0])
      gradient.append('stop')
        .attr('offset', `${t * 100}%`)
        .attr('stop-color', colorScaleFn(value))
    }

    // Add legend rectangle
    const legend = g.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${legendX},${legendY})`)

    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', `url(#${gradientId})`)

    legend.append('g')
      .attr('transform', `translate(0,${legendHeight})`)
      .call(legendAxis)
      .style('font-size', `${chartTheme.fonts.sizes.label}px`)

    // Add legend title
    legend.append('text')
      .attr('x', legendWidth / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .style('font-size', `${chartTheme.fonts.sizes.label}px`)
      .style('font-weight', 'bold')
      .text(
        colorScale === 'cooperationRate' ? 'Mutual Cooperation Rate' :
        colorScale === 'scoreDiff' ? 'Average Score Difference' :
        'Win Rate'
      )

    // Add title
    addChartTitle(svg, title, width, margin.top)

  }, [matchupData, agents, colorScale, title, showTooltip, hideTooltip, onCellClick])

  return (
    <div ref={containerRef} className="relative">
      <ChartWrapper loading={loading} error={error} minHeight={500} aspectRatio={1}>
        {({ width, height }) => (
          <svg 
            ref={svgRef} 
            width={width} 
            height={height}
            className="matchup-heatmap"
          />
        )}
      </ChartWrapper>
      {tooltip}
    </div>
  )
}