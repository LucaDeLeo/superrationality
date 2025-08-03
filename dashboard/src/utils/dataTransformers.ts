// Data transformation utilities for tournament visualizations

export interface RoundSummary {
  round: number
  cooperation_rate: number
  average_score: number
  score_variance: number
  power_distribution: {
    mean: number
    std: number
    min: number
    max: number
  }
  score_distribution: {
    min: number
    max: number
    avg: number
  }
  anonymized_games: AnonymizedGame[]
}

export interface AnonymizedGame {
  player1_id: string
  player2_id: string
  player1_action: 'C' | 'D'
  player2_action: 'C' | 'D'
  player1_score: number
  player2_score: number
}

export interface AgentCooperationData {
  agentId: string
  rounds: {
    round: number
    cooperationRate: number
    totalGames: number
  }[]
}

export interface AgentMatchupData {
  agent1Id: string
  agent2Id: string
  wins: number
  losses: number
  draws: number
  totalGames: number
  avgScoreDiff: number
  cooperationRate: number
}

export interface AgentScoreData {
  agentId: string
  rounds: {
    round: number
    score: number
    cumulativeScore: number
  }[]
}

export interface AgentPowerData {
  agentId: string
  rounds: {
    round: number
    power: number
  }[]
}

// Extract cooperation rates per agent across rounds
export const extractAgentCooperationRates = (
  roundSummaries: RoundSummary[]
): AgentCooperationData[] => {
  const agentMap = new Map<string, AgentCooperationData>()

  roundSummaries.forEach(summary => {
    const roundCooperation = new Map<string, { cooperations: number; total: number }>()

    summary.anonymized_games.forEach(game => {
      // Track player 1
      if (!roundCooperation.has(game.player1_id)) {
        roundCooperation.set(game.player1_id, { cooperations: 0, total: 0 })
      }
      const p1Data = roundCooperation.get(game.player1_id)!
      p1Data.total++
      if (game.player1_action === 'C') p1Data.cooperations++

      // Track player 2
      if (!roundCooperation.has(game.player2_id)) {
        roundCooperation.set(game.player2_id, { cooperations: 0, total: 0 })
      }
      const p2Data = roundCooperation.get(game.player2_id)!
      p2Data.total++
      if (game.player2_action === 'C') p2Data.cooperations++
    })

    // Update agent data
    roundCooperation.forEach((data, agentId) => {
      if (!agentMap.has(agentId)) {
        agentMap.set(agentId, {
          agentId,
          rounds: []
        })
      }
      
      agentMap.get(agentId)!.rounds.push({
        round: summary.round,
        cooperationRate: data.total > 0 ? data.cooperations / data.total : 0,
        totalGames: data.total
      })
    })
  })

  return Array.from(agentMap.values()).sort((a, b) => 
    a.agentId.localeCompare(b.agentId)
  )
}

// Create matchup matrix data for heat map
export const createMatchupMatrix = (
  roundSummaries: RoundSummary[]
): AgentMatchupData[] => {
  const matchupMap = new Map<string, AgentMatchupData>()

  roundSummaries.forEach(summary => {
    summary.anonymized_games.forEach(game => {
      const key = [game.player1_id, game.player2_id].sort().join('_')
      
      if (!matchupMap.has(key)) {
        matchupMap.set(key, {
          agent1Id: game.player1_id < game.player2_id ? game.player1_id : game.player2_id,
          agent2Id: game.player1_id < game.player2_id ? game.player2_id : game.player1_id,
          wins: 0,
          losses: 0,
          draws: 0,
          totalGames: 0,
          avgScoreDiff: 0,
          cooperationRate: 0
        })
      }

      const matchup = matchupMap.get(key)!
      matchup.totalGames++

      // Determine winner from perspective of agent1
      const scoreDiff = game.player1_id === matchup.agent1Id
        ? game.player1_score - game.player2_score
        : game.player2_score - game.player1_score

      if (scoreDiff > 0) matchup.wins++
      else if (scoreDiff < 0) matchup.losses++
      else matchup.draws++

      // Track cooperation
      const bothCooperated = game.player1_action === 'C' && game.player2_action === 'C'
      if (bothCooperated) {
        matchup.cooperationRate = 
          (matchup.cooperationRate * (matchup.totalGames - 1) + 1) / matchup.totalGames
      } else {
        matchup.cooperationRate = 
          (matchup.cooperationRate * (matchup.totalGames - 1)) / matchup.totalGames
      }

      // Update average score difference
      matchup.avgScoreDiff = 
        (matchup.avgScoreDiff * (matchup.totalGames - 1) + scoreDiff) / matchup.totalGames
    })
  })

  return Array.from(matchupMap.values())
}

// Calculate cumulative scores for agents
export const calculateAgentScores = (
  roundSummaries: RoundSummary[]
): AgentScoreData[] => {
  const agentScoreMap = new Map<string, AgentScoreData>()

  roundSummaries.forEach(summary => {
    const roundScores = new Map<string, number>()

    // Calculate scores for this round
    summary.anonymized_games.forEach(game => {
      roundScores.set(
        game.player1_id,
        (roundScores.get(game.player1_id) || 0) + game.player1_score
      )
      roundScores.set(
        game.player2_id,
        (roundScores.get(game.player2_id) || 0) + game.player2_score
      )
    })

    // Update cumulative scores
    roundScores.forEach((score, agentId) => {
      if (!agentScoreMap.has(agentId)) {
        agentScoreMap.set(agentId, {
          agentId,
          rounds: []
        })
      }

      const agentData = agentScoreMap.get(agentId)!
      const previousCumulative = agentData.rounds.length > 0
        ? agentData.rounds[agentData.rounds.length - 1].cumulativeScore
        : 0

      agentData.rounds.push({
        round: summary.round,
        score,
        cumulativeScore: previousCumulative + score
      })
    })
  })

  return Array.from(agentScoreMap.values()).sort((a, b) => 
    a.agentId.localeCompare(b.agentId)
  )
}

// Extract power distribution data
export const extractPowerDistribution = (
  roundData: any // Full round data with agent details
): AgentPowerData[] => {
  if (!roundData || !roundData.agents) return []

  return roundData.agents.map((agent: any) => ({
    agentId: agent.id,
    rounds: [{
      round: roundData.round,
      power: agent.power || 100 // Default power if not specified
    }]
  }))
}

// Generate tournament bracket structure
export interface BracketMatch {
  round: number
  matchId: string
  player1: string
  player2: string
  player1Action: 'C' | 'D'
  player2Action: 'C' | 'D'
  player1Score: number
  player2Score: number
  winner: string | null
}

export const generateTournamentBracket = (
  roundSummaries: RoundSummary[]
): BracketMatch[] => {
  const matches: BracketMatch[] = []

  roundSummaries.forEach(summary => {
    summary.anonymized_games.forEach((game, index) => {
      const winner = game.player1_score > game.player2_score
        ? game.player1_id
        : game.player1_score < game.player2_score
        ? game.player2_id
        : null

      matches.push({
        round: summary.round,
        matchId: `r${summary.round}_m${index}`,
        player1: game.player1_id,
        player2: game.player2_id,
        player1Action: game.player1_action,
        player2Action: game.player2_action,
        player1Score: game.player1_score,
        player2Score: game.player2_score,
        winner
      })
    })
  })

  return matches
}

// Filter data based on selected criteria
export interface FilterCriteria {
  rounds?: { min: number; max: number }
  agents?: string[]
  gameTypes?: ('all' | 'cooperate' | 'defect' | 'mixed')[]
}

// Apply visualization filters to game results
export const applyVisualizationFilters = (
  data: any[],
  filters: {
    roundRange: { min: number; max: number };
    selectedAgents: string[];
    gameType: 'all' | 'cooperate' | 'defect' | 'mixed';
  }
): any[] => {
  if (!data || data.length === 0) return [];

  return data.filter(game => {
    // Filter by round range
    if (game.round_number !== undefined) {
      if (game.round_number < filters.roundRange.min || game.round_number > filters.roundRange.max) {
        return false;
      }
    }

    // Filter by selected agents - include games where at least one selected agent is playing
    if (filters.selectedAgents.length > 0) {
      const hasSelectedAgent = 
        filters.selectedAgents.includes(game.agent1_id) || 
        filters.selectedAgents.includes(game.agent2_id);
      if (!hasSelectedAgent) {
        return false;
      }
    }

    // Filter by game type
    if (filters.gameType !== 'all') {
      const agent1Action = game.agent1_choice === 'COOPERATE' ? 'C' : 'D';
      const agent2Action = game.agent2_choice === 'COOPERATE' ? 'C' : 'D';
      
      const gameType = 
        agent1Action === 'C' && agent2Action === 'C' ? 'cooperate' :
        agent1Action === 'D' && agent2Action === 'D' ? 'defect' :
        'mixed';
      
      if (gameType !== filters.gameType) {
        return false;
      }
    }

    return true;
  });
}

export const filterRoundSummaries = (
  summaries: RoundSummary[],
  criteria: FilterCriteria
): RoundSummary[] => {
  return summaries.filter(summary => {
    // Filter by rounds
    if (criteria.rounds) {
      if (summary.round < criteria.rounds.min || summary.round > criteria.rounds.max) {
        return false
      }
    }

    // If we need to filter by agents or game types, we need to filter games
    if (criteria.agents || criteria.gameTypes) {
      const filteredGames = summary.anonymized_games.filter(game => {
        // Filter by agents
        if (criteria.agents && criteria.agents.length > 0) {
          if (!criteria.agents.includes(game.player1_id) && 
              !criteria.agents.includes(game.player2_id)) {
            return false
          }
        }

        // Filter by game type
        if (criteria.gameTypes && criteria.gameTypes.length > 0 && 
            !criteria.gameTypes.includes('all')) {
          const gameType = 
            game.player1_action === 'C' && game.player2_action === 'C' ? 'cooperate' :
            game.player1_action === 'D' && game.player2_action === 'D' ? 'defect' :
            'mixed'
          
          if (!criteria.gameTypes.includes(gameType)) {
            return false
          }
        }

        return true
      })

      // Create a new summary with filtered games
      return {
        ...summary,
        anonymized_games: filteredGames
      }
    }

    return true
  })
}