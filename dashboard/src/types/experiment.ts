export interface ExperimentSummary {
  experiment_id: string
  start_time: string
  end_time: string
  total_rounds: number
  total_games: number
  total_api_calls: number
  total_cost: number
  status: string
}

export interface ExperimentDetail extends ExperimentSummary {
  acausal_indicators: Record<string, number>
  round_count: number
  agent_count: number
}

export interface Agent {
  id: string
  power: number
  total_score: number
  games_played: number
  cooperations: number
  defections: number
  model?: string
  temperature?: number
}

export interface GameResult {
  agent1_id: string
  agent2_id: string
  agent1_choice: 'COOPERATE' | 'DEFECT'
  agent2_choice: 'COOPERATE' | 'DEFECT'
  agent1_score: number
  agent2_score: number
  outcome: string
  round_number?: number
}

export interface RoundSummary {
  round_number: number
  timestamp: string
  agents: Agent[]
  games: GameResult[]
  cooperation_rate: number
  mutual_cooperation_rate: number
  mutual_defection_rate: number
  exploitation_rate: number
}

export interface StrategyRecord {
  agent_id: string
  round_number: number
  timestamp: string
  strategy_text: string
  model_used: string
  temperature: number
  tokens_used: number
}

export interface ExperimentResult {
  experiment_id: string
  config: any
  start_time: string
  end_time: string
  total_rounds: number
  total_games: number
  total_api_calls: number
  total_cost: number
  agents: Agent[]
  round_summaries: RoundSummary[]
  acausal_indicators: Record<string, number>
  final_ranking: Agent[]
}