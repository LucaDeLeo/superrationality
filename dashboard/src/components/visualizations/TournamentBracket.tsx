import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { ChartWrapper } from './ChartWrapper';
import { ChartTooltip } from './ChartTooltip';
import { chartTheme } from './chartTheme';
import { GameResult } from '../../types';

interface BracketNode {
  agent1: string;
  agent2: string;
  round: number;
  gameId: string;
  result: GameResult;
  x: number;
  y: number;
  expanded: boolean;
}

interface TournamentBracketProps {
  data: GameResult[];
  width?: number;
  height?: number;
  onGameClick?: (game: GameResult) => void;
}

export const TournamentBracket: React.FC<TournamentBracketProps> = ({
  data,
  width = 800,
  height = 600,
  onGameClick
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: React.ReactNode } | null>(null);
  const [selectedRound, setSelectedRound] = useState<number | null>(null);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 40, right: 40, bottom: 40, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Generate bracket structure from round-robin matches
    const bracketNodes: BracketNode[] = [];
    const rounds = new Map<number, GameResult[]>();

    // Group games by round
    data.forEach(game => {
      const round = game.round || 0;
      if (!rounds.has(round)) {
        rounds.set(round, []);
      }
      rounds.get(round)!.push(game);
    });

    // Convert to bracket nodes
    let nodeId = 0;
    rounds.forEach((games, round) => {
      games.forEach((game, index) => {
        bracketNodes.push({
          agent1: game.agent1_id,
          agent2: game.agent2_id,
          round,
          gameId: `${round}-${index}`,
          result: game,
          x: 0,
          y: 0,
          expanded: expandedNodes.has(`${round}-${index}`)
        });
        nodeId++;
      });
    });

    // Calculate positions
    const roundCount = Math.max(...Array.from(rounds.keys())) + 1;
    const maxGamesPerRound = Math.max(...Array.from(rounds.values()).map(g => g.length));
    const nodeWidth = 120;
    const nodeHeight = 60;
    const horizontalSpacing = innerWidth / roundCount;
    const verticalSpacing = innerHeight / maxGamesPerRound;

    bracketNodes.forEach(node => {
      const roundGames = rounds.get(node.round)!;
      const roundIndex = roundGames.indexOf(node.result);
      node.x = node.round * horizontalSpacing + nodeWidth / 2;
      node.y = roundIndex * verticalSpacing + nodeHeight / 2;
    });

    // Draw connections between rounds
    const links = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(bracketNodes.filter(n => n.round > 0))
      .enter()
      .append('line')
      .attr('x1', d => (d.round - 1) * horizontalSpacing + nodeWidth)
      .attr('y1', innerHeight / 2)
      .attr('x2', d => d.x - nodeWidth / 2)
      .attr('y2', d => d.y)
      .attr('stroke', chartTheme.colors.grid)
      .attr('stroke-width', 1)
      .attr('opacity', 0.3);

    // Draw nodes
    const nodes = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(bracketNodes)
      .enter()
      .append('g')
      .attr('transform', d => `translate(${d.x - nodeWidth / 2},${d.y - nodeHeight / 2})`)
      .attr('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation();
        if (onGameClick) {
          onGameClick(d.result);
        }
        // Toggle expansion
        const nodeKey = d.gameId;
        const newExpanded = new Set(expandedNodes);
        if (newExpanded.has(nodeKey)) {
          newExpanded.delete(nodeKey);
        } else {
          newExpanded.add(nodeKey);
        }
        setExpandedNodes(newExpanded);
      });

    // Node backgrounds
    nodes.append('rect')
      .attr('width', nodeWidth)
      .attr('height', nodeHeight)
      .attr('rx', 4)
      .attr('fill', d => {
        if (selectedRound !== null && d.round !== selectedRound) {
          return chartTheme.colors.background;
        }
        const cooperated = d.result.action1 === 'C' && d.result.action2 === 'C';
        const defected = d.result.action1 === 'D' && d.result.action2 === 'D';
        if (cooperated) return chartTheme.colors.success + '20';
        if (defected) return chartTheme.colors.danger + '20';
        return chartTheme.colors.warning + '20';
      })
      .attr('stroke', d => {
        const cooperated = d.result.action1 === 'C' && d.result.action2 === 'C';
        const defected = d.result.action1 === 'D' && d.result.action2 === 'D';
        if (cooperated) return chartTheme.colors.success;
        if (defected) return chartTheme.colors.danger;
        return chartTheme.colors.warning;
      })
      .attr('stroke-width', 2)
      .attr('opacity', d => selectedRound !== null && d.round !== selectedRound ? 0.3 : 1);

    // Agent names
    nodes.append('text')
      .attr('x', nodeWidth / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('fill', chartTheme.colors.text)
      .text(d => d.agent1);

    nodes.append('text')
      .attr('x', nodeWidth / 2)
      .attr('y', nodeHeight - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('fill', chartTheme.colors.text)
      .text(d => d.agent2);

    // Action icons
    nodes.append('text')
      .attr('x', 20)
      .attr('y', nodeHeight / 2 + 4)
      .attr('text-anchor', 'middle')
      .attr('font-size', 16)
      .attr('font-weight', 'bold')
      .attr('fill', d => d.result.action1 === 'C' ? chartTheme.colors.success : chartTheme.colors.danger)
      .text(d => d.result.action1);

    nodes.append('text')
      .attr('x', nodeWidth - 20)
      .attr('y', nodeHeight / 2 + 4)
      .attr('text-anchor', 'middle')
      .attr('font-size', 16)
      .attr('font-weight', 'bold')
      .attr('fill', d => d.result.action2 === 'C' ? chartTheme.colors.success : chartTheme.colors.danger)
      .text(d => d.result.action2);

    // Expanded details
    const expandedDetails = nodes
      .filter(d => d.expanded)
      .append('g')
      .attr('transform', `translate(0,${nodeHeight})`);

    expandedDetails.append('rect')
      .attr('width', nodeWidth)
      .attr('height', 60)
      .attr('fill', chartTheme.colors.background)
      .attr('stroke', chartTheme.colors.border)
      .attr('stroke-width', 1);

    expandedDetails.append('text')
      .attr('x', nodeWidth / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', 11)
      .attr('fill', chartTheme.colors.textSecondary)
      .text(d => `Scores: ${d.result.score1} - ${d.result.score2}`);

    expandedDetails.append('text')
      .attr('x', nodeWidth / 2)
      .attr('y', 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', 11)
      .attr('fill', chartTheme.colors.textSecondary)
      .text(d => `Power: ${d.result.power1?.toFixed(1)} - ${d.result.power2?.toFixed(1)}`);

    // Round labels
    const roundLabels = g.append('g')
      .attr('class', 'round-labels')
      .selectAll('text')
      .data(Array.from(rounds.keys()))
      .enter()
      .append('text')
      .attr('x', d => d * horizontalSpacing + nodeWidth / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .attr('font-size', 14)
      .attr('font-weight', 'bold')
      .attr('fill', chartTheme.colors.text)
      .text(d => `Round ${d + 1}`)
      .attr('cursor', 'pointer')
      .on('click', (event, d) => {
        setSelectedRound(selectedRound === d ? null : d);
      });

    // Hover effects
    nodes
      .on('mouseenter', function(event, d) {
        d3.select(this).select('rect').attr('stroke-width', 3);
        
        const tooltipContent = (
          <div>
            <div className="font-semibold">{d.agent1} vs {d.agent2}</div>
            <div className="text-sm">Round {d.round + 1}</div>
            <div className="text-sm mt-1">
              Actions: {d.result.action1} vs {d.result.action2}
            </div>
            <div className="text-sm">
              Scores: {d.result.score1} - {d.result.score2}
            </div>
            {d.result.power1 && (
              <div className="text-sm">
                Power: {d.result.power1.toFixed(1)} - {d.result.power2?.toFixed(1)}
              </div>
            )}
          </div>
        );
        
        const bbox = (this as SVGGElement).getBoundingClientRect();
        setTooltip({
          x: bbox.left + bbox.width / 2,
          y: bbox.top - 10,
          content: tooltipContent
        });
      })
      .on('mouseleave', function() {
        d3.select(this).select('rect').attr('stroke-width', 2);
        setTooltip(null);
      });

  }, [data, width, height, selectedRound, expandedNodes, onGameClick]);

  return (
    <ChartWrapper title="Tournament Bracket" className="tournament-bracket">
      <svg ref={svgRef} width={width} height={height} />
      {tooltip && (
        <ChartTooltip
          x={tooltip.x}
          y={tooltip.y}
          content={tooltip.content}
        />
      )}
      <div className="mt-4 text-sm text-gray-600">
        Click on matches to expand details. Click on round labels to filter.
      </div>
    </ChartWrapper>
  );
};